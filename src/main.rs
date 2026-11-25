//! Уздá — минималистичный рантайм для LLM-обвязок.
//! Один бинарник, один YAML-манифест: промпты, инструменты, навыки, лимиты
//! и граф этапов исполняются против OpenAI-совместимого API.
//!
//!   uzda run   harness.yaml ["запрос"] [--stage S] [--json]
//!   uzda serve harness.yaml [адрес]    # POST /v1/chat/completions, GET /health

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

// ─── Манифест ────────────────────────────────────────────────────────────────

#[derive(Clone, Deserialize)]
struct Model { url: String, name: String, api_key: Option<String> }

#[derive(Clone, Deserialize)]
struct PromptDef { name: String, path: String, policy: Option<String> }

#[derive(Clone, Deserialize)]
struct ToolDef { name: String, run: String, desc: String, timeout: Option<u64>, args: Option<Value> }

#[derive(Clone, Deserialize)]
struct SkillDef { name: String, desc: String, instr: String }

#[derive(Clone, Deserialize)]
struct LoopLimit { max_repeat: Option<usize>, window: Option<usize> }

#[derive(Clone, Deserialize)]
struct Limits {
    turns: Option<usize>, time: Option<u64>, cost: Option<f64>,
    #[serde(rename = "loop")] loop_limit: Option<LoopLimit>,
    context: Option<HashMap<String, Value>>, // window | full_at
}

fn ctx_get(l: &Option<Limits>, key: &str) -> Option<Value> {
    l.as_ref()?.context.as_ref()?.get(key).cloned()
}

#[derive(Clone, Deserialize)]
struct Stage {
    name: String, desc: String,
    prompt: Option<Vec<String>>, tools: Option<Vec<String>>,
    skills: Option<Vec<String>>, next: Option<Vec<String>>,
    history: Option<String>,
}

#[derive(Clone, Deserialize)]
struct Manifest {
    model: Model,
    #[serde(default)] prompts: Vec<PromptDef>,
    #[serde(default)] tools: Vec<ToolDef>,
    #[serde(default)] skills: Vec<SkillDef>,
    #[serde(default)] limits: Option<Limits>,
    #[serde(default)] stages: Vec<Stage>,
}

// ─── Сообщения и запрос к провайдеру ─────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
struct Call { id: String, #[serde(rename = "type")] kind: String, function: Fn_ }

#[derive(Serialize, Deserialize, Clone)]
struct Fn_ { name: String, arguments: String }

#[derive(Serialize, Deserialize, Clone)]
struct Msg {
    role: String, content: String,
    #[serde(skip_serializing_if = "Option::is_none", default)] tool_calls: Option<Vec<Call>>,
    #[serde(skip_serializing_if = "Option::is_none", default)] tool_call_id: Option<String>,
}

fn user(s: &str) -> Msg { Msg { role: "user".into(), content: s.into(), tool_calls: None, tool_call_id: None } }

// ─── Состояние прогона ────────────────────────────────────────────────────────

struct Run<'a> {
    m: &'a Manifest, http: reqwest::Client,
    sys: String, stage: Option<&'a Stage>, tools: Vec<&'a ToolDef>, skills: Vec<&'a SkillDef>,
    hist: Vec<Msg>, sigs: Vec<String>, turns: usize, spent: f64, t0: Instant,
    deadline: u64, max_turns: usize, budget: f64, max_rep: usize, win: usize,
    ctx_win: usize, full_at: f64,
}

fn read_file(p: &str) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("нет файла '{p}': {e}"))
}

impl<'a> Run<'a> {
    fn new(m: &'a Manifest, hist: Vec<Msg>) -> Run<'a> {
        let l = &m.limits;
        Run {
            m, http: reqwest::Client::new(),
            sys: String::new(), stage: None, tools: Vec::new(), skills: Vec::new(),
            hist, sigs: Vec::new(), turns: 0, spent: 0.0, t0: Instant::now(),
            deadline: l.as_ref().and_then(|x| x.time).unwrap_or(300),
            max_turns: l.as_ref().and_then(|x| x.turns).unwrap_or(30),
            budget: l.as_ref().and_then(|x| x.cost).unwrap_or(f64::MAX),
            max_rep: l.as_ref().and_then(|x| x.loop_limit.as_ref()).and_then(|x| x.max_repeat).unwrap_or(3),
            win: l.as_ref().and_then(|x| x.loop_limit.as_ref()).and_then(|x| x.window).unwrap_or(10),
            ctx_win: ctx_get(l, "window").as_ref().and_then(|v| v.as_u64()).unwrap_or(48000) as usize,
            full_at: ctx_get(l, "full_at").as_ref().and_then(|v| v.as_f64()).unwrap_or(0.9),
        }
    }

    /// Вход в этап: собрать system из промптов, ограничить инструменты/навыки.
    /// Пустой список (`[]`) — явная очистка; отсутствие поля — наследование глобальной секции.
    fn enter(&mut self, name: &str) -> Result<(), String> {
        let st = self.m.stages.iter().find(|s| s.name == name)
            .ok_or_else(|| format!("нет этапа '{name}'"))?;
        let names: Vec<String> = match &st.prompt {
            Some(list) => list.clone(),
            None => {
                let mut v: Vec<String> = self.m.prompts.iter()
                    .filter(|p| p.policy.as_deref() != Some("slide")).map(|p| p.name.clone()).collect();
                v.extend(self.m.prompts.iter().filter(|p| p.policy.as_deref() == Some("slide")).map(|p| p.name.clone()));
                v
            }
        };
        let mut sys = names.iter()
            .filter_map(|n| self.m.prompts.iter().find(|p| p.name == *n))
            .map(|p| read_file(&p.path)).collect::<Vec<_>>().join("\n\n");
        if self.m.stages.len() > 1 {
            let g = st.next.as_ref().map(Vec::as_slice).unwrap_or(&[]);
            if !g.is_empty() {
                let rows: Vec<String> = self.m.stages.iter().filter(|s| g.contains(&s.name))
                    .map(|s| format!("- {}: {}", s.name, s.desc)).collect();
                sys.push_str(&format!(
                    "\n\nДОСТУПНЫЕ ЭТАПЫ (переход вызовом go):\n{}\nЕсли задача решена — ответь текстом без вызова инструментов.",
                    rows.join("\n")));
            } else {
                sys.push_str("\n\nЭто завершающий этап: ответь итоговым текстом без вызова инструментов.");
            }
        }
        self.sys = sys;
        self.stage = Some(st);
        self.tools = match &st.tools {
            Some(list) => list.iter().filter_map(|n| self.m.tools.iter().find(|t| t.name == *n)).collect(),
            None => self.m.tools.iter().collect(),
        };
        self.skills = match &st.skills {
            Some(list) => list.iter().filter_map(|n| self.m.skills.iter().find(|s| s.name == *n)).collect(),
            None => self.m.skills.iter().collect(),
        };
        Ok(())
    }

    /// Массив tools для провайдера: команды этапа + навыки + переходы графа.
    fn tool_specs(&self) -> Vec<Value> {
        let simple = json!({"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]});
        let mut v: Vec<Value> = self.tools.iter().map(|t| json!({
            "type": "function", "function": {
                "name": t.name, "description": t.desc,
                "parameters": t.args.clone().unwrap_or_else(|| simple.clone())
            }})).collect();
        v.extend(self.skills.iter().map(|s| json!({
            "type": "function", "function": {
                "name": s.name, "description": s.desc, "parameters": simple.clone()
            }})));
        if let Some(st) = self.stage {
            if let Some(next) = &st.next {
                if !next.is_empty() {
                    v.push(json!({"type": "function", "function": {
                        "name": "go", "description": "Перейти на другой этап работы.",
                        "parameters": {"type": "object", "properties": {"stage": {"type": "string", "enum": next}},
                                      "required": ["stage"]}
                    }}));
                }
            }
        }
        v
    }

    /// Запрос к провайдеру. Возвращает сообщение ассистента и usage.
    async fn chat(&self, extra: &[Msg]) -> Result<(Msg, u64, u64), String> {
        let brief = self.stage.and_then(|s| s.history.as_deref()) == Some("brief");
        let mut msgs = vec![Msg { role: "system".into(), content: self.sys.clone(), tool_calls: None, tool_call_id: None }];
        if brief {
            for m in self.hist.iter().filter(|m| m.role != "tool" && m.tool_calls.is_none()) {
                let mut c = m.clone();
                c.content = c.content.chars().take(240).collect();
                msgs.push(c);
            }
        } else {
            msgs.extend_from_slice(&self.hist);
        }
        msgs.extend_from_slice(extra);
        let body = json!({"model": self.m.model.name, "messages": msgs, "tools": self.tool_specs()});
        let url = format!("{}/chat/completions", self.m.model.url.trim_end_matches('/'));
        let resp: Value = self.http.post(&url)
            .bearer_auth(self.m.model.api_key.clone().unwrap_or_else(|| "none".into()))
            .json(&body).send().await.map_err(|e| format!("провайдер: {e}"))?
            .error_for_status().map_err(|e| format!("провайдер: {e}"))?
            .json().await.map_err(|e| format!("провайдер: {e}"))?;
        let u = &resp["usage"];
        let out = &resp["choices"][0]["message"];
        let msg = Msg {
            role: "assistant".into(),
            content: out["content"].as_str().unwrap_or("").to_string(),
            tool_calls: out["tool_calls"].as_array().map(|cs| cs.iter().map(|c| Call {
                id: c["id"].as_str().unwrap_or_default().into(),
                kind: "function".into(),
                function: Fn_ {
                    name: c["function"]["name"].as_str().unwrap_or_default().into(),
                    arguments: c["function"]["arguments"].as_str().unwrap_or("{}").into(),
                },
            }).collect()),
            tool_call_id: None,
        };
        Ok((msg, u["prompt_tokens"].as_u64().unwrap_or(0), u["completion_tokens"].as_u64().unwrap_or(0)))
    }

    /// Исполнить один вызов инструмента/навыка.
    async fn exec(&self, c: &Fn_) -> String {
        let arg = |key: &str| -> String {
            serde_json::from_str::<Value>(&c.arguments).ok()
                .and_then(|v| v.get(key).map(|x| match x {
                    Value::String(s) => s.clone(), o => o.to_string(),
                })).unwrap_or_default()
        };
        if let Some(s) = self.skills.iter().find(|s| s.name == c.name) {
            return read_file(&s.instr); // навык: подставляем инструкцию как результат инструмента
        }
        let Some(t) = self.tools.iter().find(|t| t.name == c.name) else {
            return format!("неизвестный инструмент '{}'", c.name);
        };
        let mut cmd = Command::new(&t.run);
        for w in arg("input").split_whitespace() { cmd.arg(w); }
        cmd.kill_on_drop(true);
        let child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(x) => x, Err(e) => return format!("ошибка запуска '{}': {e}", t.run),
        };
        let res = tokio::select! {
            r = child.wait_with_output() => r,
            _ = tokio::time::sleep(Duration::from_secs(t.timeout.unwrap_or(30))) => {
                return "таймаут инструмента".into(); // kill_on_drop убирает процесс за собой
            }
        };
        match res {
            Ok(o) => {
                let mut s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() { s = String::from_utf8_lossy(&o.stderr).trim().to_string(); }
                s.chars().take(8000).collect()
            }
            Err(e) => format!("ошибка: {e}"),
        }
    }

    /// Скользящее окно: при заполнении контекста старые ходы сворачиваются в краткую запись.
    fn compact(&mut self) {
        let est = (self.sys.len() + self.hist.iter().map(|m| m.content.len()).sum::<usize>()) / 4;
        if est < (self.ctx_win as f64 * self.full_at) as usize || self.hist.len() <= 10 { return; }
        let mut cut = self.hist.len() - 8;
        while cut > 0 && self.hist[cut].role == "tool" { cut -= 1; }
        if cut == 0 { return; }
        let old: Vec<Msg> = self.hist.drain(..cut).collect();
        let rows: Vec<String> = old.iter().filter(|m| m.role != "tool")
            .map(|m| format!("[{}] {}", m.role, m.content.chars().take(120).collect::<String>())).collect();
        self.hist.insert(0, user(&format!("КРАТКО ПРЕДЫСТОРИИ:\n{}", rows.join("\n"))));
    }

    /// Один шаг: запрос к модели + исполнение вызовов. Возвращает новые сообщения истории.
    async fn step(&mut self, extra: &[Msg]) -> Result<Vec<Msg>, String> {
        if self.t0.elapsed().as_secs() >= self.deadline { return Err("превышено время прогона".into()); }
        if self.spent >= self.budget { return Err("превышен бюджет".into()); }
        self.compact();
        let (mut msg, pt, ct) = self.chat(extra).await?;
        self.turns += 1;
        self.spent += (pt as f64 * 15.0 + ct as f64 * 75.0) / 1e6; // $/Мтокен
        let calls = msg.tool_calls.take().unwrap_or_default();
        let mut out = vec![msg];
        if calls.is_empty() || self.turns >= self.max_turns { return Ok(out); }
        for c in &calls {
            if c.function.name == "go" {
                let to = serde_json::from_str::<BTreeMap<String, String>>(&c.function.arguments)
                    .ok().and_then(|a| a.get("stage").cloned()).unwrap_or_default();
                self.enter(&to)?;
                out.push(Msg { role: "tool".into(), content: format!("этап: {to}"), tool_calls: None, tool_call_id: Some(c.id.clone()) });
                continue;
            }
            let sig = format!("{} {}", c.function.name, c.function.arguments);
            self.sigs.push(sig.clone());
            if self.sigs.iter().filter(|x| **x == sig).count() >= self.max_rep {
                return Err(format!("зацикливание: '{}' повторён {} раз", c.function.name, self.max_rep));
            }
            if self.sigs.len() > self.win { self.sigs.remove(0); }
            let res = self.exec(&c.function).await;
            out.push(Msg { role: "tool".into(), content: res, tool_calls: None, tool_call_id: Some(c.id.clone()) });
        }
        Ok(out)
    }

    /// Полный прогон одного вопроса.
    async fn drive(&mut self, q: &str) -> Result<String, String> {
        let first = user(q);
        let mut pending = self.step(std::slice::from_ref(&first)).await?;
        self.hist.push(first);
        loop {
            let has_tool = pending.iter().any(|m| m.role == "tool");
            self.hist.append(&mut pending);
            let text = self.hist.last().map(|m| m.content.clone()).unwrap_or_default();
            if !has_tool || self.turns >= self.max_turns { return Ok(text); }
            pending = self.step(&[]).await?;
        }
    }
}

// ─── Манифест: загрузка и резолв путей ───────────────────────────────────────

fn load(path: &str) -> Result<Manifest, String> {
    let p = PathBuf::from(path);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new(".")).to_path_buf();
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{path}: {e}"))?;
    let mut m: Manifest = serde_yaml::from_str(&text).map_err(|e| format!("YAML: {e}"))?;
    for pr in &mut m.prompts { pr.path = dir.join(&pr.path).to_string_lossy().into(); }
    for s in &mut m.skills { s.instr = dir.join(&s.instr).to_string_lossy().into(); }
    if let Ok(v) = std::env::var("UZDA_MODEL_URL") { if !v.is_empty() { m.model.url = v; } }
    if let Ok(v) = std::env::var("UZDA_MODEL_NAME") { if !v.is_empty() { m.model.name = v; } }
    Ok(m)
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let code = match a.first().map(String::as_str) {
        Some("run") if a.len() > 1 => run_cli(&a[1..]).await,
        Some("serve") if a.len() > 1 => serve(&a[1..]).await,
        _ => { eprintln!("uzda run   <harness.yaml> [\"запрос\"] [--stage S] [--json]\nuzda serve <harness.yaml> [адрес]   # по умолчанию 127.0.0.1:8080"); 2 }
    };
    std::process::exit(code);
}

async fn run_cli(a: &[String]) -> i32 {
    let flag = |n: &str| a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned();
    let query = a.get(1).filter(|s| !s.starts_with("--")).cloned().or_else(|| {
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_ok() && !s.trim().is_empty() { Some(s.trim().to_string()) } else { None }
    });
    let m = match load(&a[0]) { Ok(x) => x, Err(e) => { eprintln!("{e}"); return 1 } };
    let mut r = Run::new(&m, Vec::new());
    let stage = flag("--stage").unwrap_or_else(|| m.stages.first().map(|s| s.name.clone()).unwrap_or_default());
    if !m.stages.is_empty() { if let Err(e) = r.enter(&stage) { eprintln!("{e}"); return 1 } }
    let result = match &query {
        Some(q) => r.drive(q).await,
        None => repl(&mut r).await,
    };
    match result {
        Ok(t) => {
            if a.iter().any(|x| x == "--json") {
                println!("{}", json!({"answer": t, "turns": r.turns, "cost": r.spent, "stage": r.stage.map(|s| s.name.clone())}));
            } else if !t.is_empty() { println!("{t}"); }
            0
        }
        Err(e) => { eprintln!("Стоп: {e}"); 1 }
    }
}

async fn repl(r: &mut Run<'_>) -> Result<String, String> {
    let stdin = std::io::stdin();
    loop {
        print!("> "); let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 || line.trim() == "/exit" { return Ok(String::new()); }
        if line.trim().is_empty() { continue; }
        match r.drive(line.trim()).await { Ok(t) => println!("{t}"), Err(e) => eprintln!("Стоп: {e}") }
    }
}

// ─── Serve: минимальный OpenAI-совместимый API поверх агента ─────────────────

/// Сессия = (история, текущий этап). Живёт в памяти процесса и умирает вместе с ним.
type Sessions = std::sync::Arc<tokio::sync::Mutex<HashMap<String, (Vec<Msg>, String)>>>;

async fn serve(a: &[String]) -> i32 {
    let addr = a.get(1).cloned().unwrap_or_else(|| "127.0.0.1:8080".into());
    let m = match load(&a[0]) { Ok(x) => x, Err(e) => { eprintln!("{e}"); return 1 } };
    let m = std::sync::Arc::new(m);
    let sessions: Sessions = Default::default();
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l, Err(e) => { eprintln!("bind {addr}: {e}"); return 1 }
    };
    eprintln!("uzda → http://{addr}/v1/chat/completions ({})", m.model.name);
    let mut shutdown = std::pin::pin!(tokio::signal::ctrl_c());
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let stream = match accepted { Ok((s, _)) => s, Err(_) => continue };
                let (m, s) = (m.clone(), sessions.clone());
                tokio::spawn(async move { let _ = conn(stream, m, s).await; });
            }
        }
    }
    0
}

/// Обработка одного HTTP-соединения вручную: один GET /health, один POST /v1/chat/completions.
async fn conn(mut sock: tokio::net::TcpStream, m: std::sync::Arc<Manifest>, sessions: Sessions) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = find_sub(&buf, b"\r\n\r\n") { break i + 4; }
        match sock.read(&mut tmp).await { Ok(0) | Err(_) => return, Ok(n) => buf.extend_from_slice(&tmp[..n]) }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.lines();
    let mut it = lines.next().unwrap_or("").split(' ');
    let (method, path) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
    let clen = head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim().eq_ignore_ascii_case("content-length") { v.trim().parse::<usize>().ok() } else { None }
    }).unwrap_or(0);
    while buf.len() < head_end + clen {
        match sock.read(&mut tmp).await { Ok(0) | Err(_) => return, Ok(n) => buf.extend_from_slice(&tmp[..n]) }
    }
    let body = String::from_utf8_lossy(&buf[head_end..head_end + clen]).to_string();
    let resp = match (method, path) {
        ("GET", "/health") => reply(200, "text/plain", "ok"),
        ("POST", "/v1/chat/completions") => api(body, &m, sessions).await,
        _ => reply(404, "application/json", "{\"error\":\"неизвестный путь\"}"),
    };
    let _ = sock.write_all(resp.as_bytes()).await;
    let _ = sock.flush().await;
}

async fn api(body: String, m: &Manifest, sessions: Sessions) -> String {
    let json_err = |code: u16, e: &str| reply(code, "application/json", &json!({"error": e}).to_string());
    let Ok(req) = serde_json::from_str::<Value>(&body) else { return json_err(400, "нужен JSON тела запроса") };
    let Ok(msgs) = serde_json::from_value::<Vec<Msg>>(req["messages"].clone()) else { return json_err(400, "нужен массив messages") };
    let key = req.get("session").and_then(Value::as_str).unwrap_or("default").to_string();
    let (hist, stage) = {
        let g = sessions.lock().await;
        g.get(&key).cloned().unwrap_or_else(|| (Vec::new(), m.stages.first().map(|s| s.name.clone()).unwrap_or_default()))
    };
    let last = msgs.iter().rev().find(|x| x.role == "user").map(|x| x.content.clone()).unwrap_or_default();
    let mut r = Run::new(m, hist);
    if !m.stages.is_empty() && r.enter(&stage).is_err() { return json_err(400, "нет такого этапа"); }
    match r.drive(&last).await {
        Ok(t) => {
            let cur = r.stage.map(|s| s.name.clone()).unwrap_or_default();
            sessions.lock().await.insert(key, (r.hist, cur));
            reply(200, "application/json", &json!({
                "id": format!("uzda-{}", millis()), "object": "chat.completion", "model": m.model.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }).to_string())
        }
        Err(e) => json_err(422, &e),
    }
}

fn reply(code: u16, ctype: &str, body: &str) -> String {
    let text = match code { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", _ => "Unprocessable Entity" };
    format!("HTTP/1.1 {code} {text}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn millis() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}
