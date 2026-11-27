//! Uzda: a minimal runtime. Manifest -> model -> trace. One binary, stateless by default.
#![forbid(unsafe_code)]

/// A poisoned lock means a panic elsewhere, not corrupt data: the maps stay consistent.
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ===== Runtime budgets =====
mod budget {
    use std::time::Duration;

    pub const MAX_OUTPUT: usize = 32 * 1024;
    pub const MAX_CALLS: usize = 16;
    pub const MAX_QUERY: usize = 64 * 1024;
    pub const MAX_TRACE: u64 = 8 * 1024 * 1024;
    pub const TOOL_SLOTS: usize = 8;
    pub const REPLAY: usize = 256;
    pub const CONTEXT_CHARS: usize = 128 * 1024;
    pub const ID_TRIES: usize = 64;
    pub const MODEL_TIMEOUT: Duration = Duration::from_secs(120);
    pub const LISTING_TTL: Duration = Duration::from_secs(2);
}

// ===== Errors: one vocabulary for the whole HTTP boundary =====
mod fail {
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};

    pub enum Error {
        BadRequest(String),
        Unauthorized,
        Forbidden,
        NotFound,
        Conflict,
        TooLarge,
        Invalid(String),
        Io(std::io::Error),
    }

    pub type Result<T> = std::result::Result<T, Error>;

    impl From<std::io::Error> for Error {
        fn from(e: std::io::Error) -> Self {
            Self::Io(e)
        }
    }

    impl IntoResponse for Error {
        fn into_response(self) -> Response {
            let (status, message) = match self {
                Self::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
                Self::Unauthorized => (StatusCode::UNAUTHORIZED, "token required".to_string()),
                Self::Forbidden => (StatusCode::FORBIDDEN, "host not allowed".to_string()),
                Self::NotFound => (StatusCode::NOT_FOUND, "not found".to_string()),
                Self::Conflict => (
                    StatusCode::CONFLICT,
                    "session is already active".to_string(),
                ),
                Self::TooLarge => (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "payload too large".to_string(),
                ),
                Self::Invalid(m) => (StatusCode::UNPROCESSABLE_ENTITY, m),
                Self::Io(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    (StatusCode::NOT_FOUND, "not found".to_string())
                }
                Self::Io(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            };
            (status, message).into_response()
        }
    }
}

// ===== YAML emission =====
mod yaml {
    fn plain_ok(s: &str) -> bool {
        if s.is_empty() || s.trim() != s || s.contains('\n') {
            return false;
        }
        if s.chars()
            .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
        {
            return false;
        }
        if s.starts_with([
            '-', '?', ':', ',', '[', ']', '{', '}', '#', '&', '*', '!', '|', '>', '\'', '"', '%',
            '@', '`', '.', '=',
        ]) {
            return false;
        }
        if s.contains(": ") || s.ends_with(':') || s.contains(" #") {
            return false;
        }
        if s.as_bytes()
            .windows(2)
            .any(|w| w[0] == b':' && w[1].is_ascii_digit())
        {
            return false;
        }
        let low = s.to_ascii_lowercase();
        if matches!(
            low.as_str(),
            "true"
                | "false"
                | "null"
                | "yes"
                | "no"
                | "on"
                | "off"
                | "y"
                | "n"
                | "~"
                | ".inf"
                | "-.inf"
                | ".nan"
        ) {
            return false;
        }
        if low.starts_with("0x") || low.starts_with("0o") || low.starts_with("0b") {
            return false;
        }
        if s.parse::<f64>().is_ok() || s.chars().all(|c| c.is_ascii_digit() || c == '_') {
            return false;
        }
        true
    }

    pub fn quote(s: &str) -> String {
        let j = serde_json::to_string(s).expect("a string always serializes");
        let special = |c: char| matches!(c, '\u{7f}'..='\u{9f}' | '\u{2028}' | '\u{2029}');
        if !j.chars().any(special) {
            return j;
        }
        let mut out = String::with_capacity(j.len());
        for c in j.chars() {
            if special(c) {
                out.push_str(&format!("\\u{:04x}", c as u32));
            } else {
                out.push(c);
            }
        }
        out
    }

    pub fn scalar(s: &str, indent: usize) -> String {
        if plain_ok(s) {
            return s.to_owned();
        }
        let clean = !s.chars().any(|c| {
            (c.is_control() && c != '\n' && c != '\t') || matches!(c, '\u{2028}' | '\u{2029}')
        });
        if s.contains('\n') && clean && !s.starts_with([' ', '\t', '\n']) {
            let (head, body) = match s.strip_suffix('\n') {
                Some(b) if b.ends_with('\n') => ("|+", b),
                Some(b) => ("|", b),
                None => ("|-", s),
            };
            let pad = " ".repeat(indent);
            let mut out = head.to_owned();
            for line in body.split('\n') {
                out.push('\n');
                out.push_str(&pad);
                out.push_str(line);
            }
            out
        } else {
            quote(s)
        }
    }
}

// ===== Manifest =====
mod manifest {
    use anyhow::bail;
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::time::Duration;

    #[derive(Deserialize)]
    pub struct Manifest {
        pub model: Model,
        #[serde(default)]
        pub prompts: Vec<PromptDef>,
        #[serde(default)]
        pub messages: Vec<MessageDef>,
        #[serde(default)]
        pub skills: Vec<SkillDef>,
        #[serde(default)]
        pub tools: Vec<ToolDef>,
        #[serde(default)]
        pub stages: Vec<StageDef>,
        #[serde(default)]
        pub limits: Limits,
    }

    #[derive(Deserialize)]
    pub struct Model {
        pub url: String,
        #[serde(default = "default_name")]
        pub name: String,
    }

    fn default_name() -> String {
        "default".to_string()
    }

    #[derive(Deserialize)]
    pub struct PromptDef {
        pub name: String,
        pub path: String,
        #[serde(default = "pin")]
        pub policy: String,
    }

    fn pin() -> String {
        "pin".to_string()
    }

    #[derive(Deserialize)]
    pub struct MessageDef {
        pub name: String,
        pub path: String,
    }

    #[derive(Deserialize)]
    pub struct SkillDef {
        pub name: String,
        pub instr: String,
        #[serde(default)]
        pub desc: String,
    }

    #[derive(Deserialize)]
    pub struct ToolDef {
        pub name: String,
        pub run: String,
        #[serde(default)]
        pub desc: String,
        #[serde(default = "default_timeout", deserialize_with = "de_timeout")]
        pub timeout: Duration,
        #[serde(default)]
        pub schema: Option<String>,
        #[serde(default)]
        pub shell: bool,
        #[serde(default)]
        pub uid: Option<u32>,
        #[serde(default)]
        pub gid: Option<u32>,
    }

    fn default_timeout() -> Duration {
        Duration::from_secs(30)
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TimeoutRepr {
        Secs(u64),
        Text(String),
    }

    fn de_timeout<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let text = match TimeoutRepr::deserialize(d)? {
            TimeoutRepr::Secs(n) => return Ok(Duration::from_secs(n)),
            TimeoutRepr::Text(s) => s,
        };
        parse_dur(&text).map_err(serde::de::Error::custom)
    }

    fn parse_dur(s: &str) -> Result<Duration, String> {
        let s = s.trim();
        let (num, mul) = if let Some(n) = s.strip_suffix("ms") {
            (n, 1)
        } else if let Some(n) = s.strip_suffix('s') {
            (n, 1000)
        } else if let Some(n) = s.strip_suffix('m') {
            (n, 60_000)
        } else {
            (s, 1000)
        };
        num.trim()
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(mul))
            .map(Duration::from_millis)
            .ok_or_else(|| format!("invalid duration \"{s}\""))
    }

    #[derive(Clone, Copy, Default, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Kind {
        #[default]
        #[serde(rename = "string")]
        Str,
        Number,
        Integer,
        Boolean,
    }

    impl Kind {
        pub fn label(self) -> &'static str {
            match self {
                Self::Str => "string",
                Self::Number => "number",
                Self::Integer => "integer",
                Self::Boolean => "boolean",
            }
        }

        pub fn matches(self, v: &serde_json::Value) -> bool {
            match self {
                Self::Str => v.is_string(),
                Self::Number => v.is_number(),
                Self::Integer => v.as_i64().is_some() || v.as_u64().is_some(),
                Self::Boolean => v.is_boolean(),
            }
        }
    }

    #[derive(Clone, Deserialize)]
    pub struct Param {
        pub name: String,
        #[serde(default)]
        pub desc: String,
        #[serde(rename = "type", default)]
        pub kind: Kind,
        #[serde(default = "yes")]
        pub required: bool,
        #[serde(default)]
        pub default: Option<serde_json::Value>,
    }

    fn yes() -> bool {
        true
    }

    #[derive(Deserialize)]
    pub struct ToolSchema {
        pub params: Vec<Param>,
        #[serde(default)]
        pub argv: Option<Vec<String>>,
    }

    #[derive(Deserialize)]
    pub struct StageDef {
        pub name: String,
        pub desc: String,
        #[serde(default)]
        pub prompt: Vec<String>,
        #[serde(default)]
        pub tools: Vec<String>,
        #[serde(default)]
        pub skills: Vec<String>,
        #[serde(default)]
        pub next: Vec<String>,
        #[serde(default)]
        pub history: Option<HistorySpec>,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum HistorySpec {
        Alias(String),
        Policy {
            #[serde(default)]
            r#use: Option<String>,
            #[serde(default)]
            mode: Option<String>,
            #[serde(default)]
            turns: Option<usize>,
            #[serde(default)]
            percent: Option<usize>,
        },
    }

    #[derive(Deserialize, Clone, Copy)]
    pub struct Limits {
        #[serde(default = "default_turns")]
        pub turns: u64,
        #[serde(default)]
        pub time: Option<u64>,
    }

    fn default_turns() -> u64 {
        30
    }

    impl Default for Limits {
        fn default() -> Self {
            Self {
                turns: default_turns(),
                time: None,
            }
        }
    }

    pub fn valid_name(n: &str) -> bool {
        !n.is_empty()
            && n.len() <= 64
            && n.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    }

    pub fn uniq<'a, T>(
        items: &'a [T],
        name: impl Fn(&'a T) -> &'a str,
        section: &str,
    ) -> anyhow::Result<HashMap<&'a str, &'a T>> {
        let mut map = HashMap::new();
        for it in items {
            let n = name(it);
            if !valid_name(n) {
                bail!("{section}: name \"{n}\" must be ascii letters, digits, _ or - (1..64)");
            }
            if map.insert(n, it).is_some() {
                bail!("{section}: duplicate name \"{n}\"");
            }
        }
        Ok(map)
    }

    pub fn refs<V>(names: &[String], known: &HashMap<&str, V>, field: &str) -> anyhow::Result<()> {
        for n in names {
            if !known.contains_key(n.as_str()) {
                bail!("{field}: unknown name \"{n}\"");
            }
        }
        Ok(())
    }
}

// ===== Trace =====
mod trace {
    use crate::budget::{ID_TRIES, MAX_TRACE};
    use crate::fail::{self, Error};
    use crate::yaml;
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;
    use tokio::fs::OpenOptions;
    use tokio::io::AsyncWriteExt;

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Entry {
        User(UserText),
        Model(String),
        Call(String),
        Result(String),
        System(String),
        Stage(String),
    }

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum UserText {
        Plain(String),
        From { from: String, text: String },
    }

    impl Entry {
        pub fn text(&self) -> &str {
            match self {
                Self::User(UserText::Plain(t)) | Self::User(UserText::From { text: t, .. }) => t,
                Self::Model(t)
                | Self::Call(t)
                | Self::Result(t)
                | Self::System(t)
                | Self::Stage(t) => t,
            }
        }

        pub fn yaml(&self) -> String {
            match self {
                Self::User(UserText::Plain(t)) => format!("- user: {}", yaml::scalar(t, 4)),
                Self::User(UserText::From { from, text }) => {
                    format!(
                        "- user:\n    from: {}\n    text: {}",
                        yaml::scalar(from, 6),
                        yaml::scalar(text, 6)
                    )
                }
                Self::Model(t) => format!("- model: {}", yaml::scalar(t, 4)),
                Self::Call(t) => format!("- call: {}", yaml::scalar(t, 4)),
                Self::Result(t) => format!("- result: {}", yaml::scalar(t, 4)),
                Self::System(t) => format!("- system: {}", yaml::scalar(t, 4)),
                Self::Stage(t) => format!("- stage: {}", yaml::scalar(t, 4)),
            }
        }
    }

    pub fn valid_id(id: &str) -> bool {
        id.len() == 8
            && id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    #[derive(Deserialize)]
    #[serde(try_from = "String")]
    pub struct TraceId(pub String);

    impl TryFrom<String> for TraceId {
        type Error = String;

        fn try_from(s: String) -> Result<Self, String> {
            if valid_id(&s) {
                Ok(Self(s))
            } else {
                Err("id: expected 8 hex chars".to_string())
            }
        }
    }

    fn fresh_id() -> String {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};
        let mut h = RandomState::new().build_hasher();
        h.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        );
        format!("{:08x}", h.finish() as u32)
    }

    pub fn traces_dir() -> std::io::Result<PathBuf> {
        let home = std::env::var("HOME")
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "HOME is not set"))?;
        Ok(PathBuf::from(home).join(".uzda").join("traces"))
    }

    pub fn trace_path(id: &str) -> fail::Result<PathBuf> {
        Ok(traces_dir()?.join(format!("{id}.yaml")))
    }

    pub async fn read_entries(id: &str) -> fail::Result<Vec<Entry>> {
        let path = trace_path(id)?;
        if tokio::fs::metadata(&path).await?.len() > MAX_TRACE {
            return Err(Error::TooLarge);
        }
        let text = tokio::fs::read_to_string(&path).await?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        serde_saphyr::from_str(&text).map_err(|e| Error::Invalid(format!("trace {id}: {e}")))
    }

    pub struct Trace {
        pub id: String,
        pub file: tokio::fs::File,
        pub entries: Vec<Entry>,

        pub broken: bool,
    }

    impl Trace {
        pub async fn create() -> fail::Result<Trace> {
            let dir = traces_dir()?;
            tokio::fs::create_dir_all(&dir).await?;
            for _ in 0..ID_TRIES {
                let id = fresh_id();
                let path = dir.join(format!("{id}.yaml"));
                match OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .await
                {
                    Ok(file) => {
                        return Ok(Trace {
                            id,
                            file,
                            entries: Vec::new(),

                            broken: false,
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => return Err(e.into()),
                }
            }
            Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "cannot allocate a trace id",
            )
            .into())
        }

        pub async fn open(id: String) -> fail::Result<Trace> {
            let entries = read_entries(&id).await?;
            let file = OpenOptions::new()
                .append(true)
                .open(trace_path(&id)?)
                .await?;
            Ok(Trace {
                id,
                file,
                entries,

                broken: false,
            })
        }

        pub async fn push(&mut self, e: Entry) -> std::io::Result<()> {
            let line = format!("{}\n", e.yaml());
            let size = self.file.metadata().await?.len();
            if size.saturating_add(line.len() as u64) > MAX_TRACE {
                return Err(std::io::Error::other("trace size limit exceeded"));
            }
            self.file.write_all(line.as_bytes()).await?;
            self.file.flush().await?;
            self.entries.push(e);
            Ok(())
        }

        pub fn stage(&self, harness: &crate::harness::Harness) -> String {
            let mut stage = harness.start.clone();
            for e in &self.entries {
                if let Entry::Stage(n) = e
                    && harness.stages.contains_key(n)
                {
                    stage = n.to_owned();
                }
            }
            stage
        }
    }
}

// ===== Harness =====
mod harness {
    use crate::budget::CONTEXT_CHARS;
    use crate::manifest::{
        HistorySpec, Manifest, Param, StageDef, ToolSchema, refs, uniq, valid_name,
    };
    use crate::trace::Entry;
    use anyhow::{Context, bail};
    use serde_json::{Map, Value, json};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    pub struct Harness {
        pub dir: PathBuf,
        pub endpoint: String,
        pub model: String,
        pub key: String,
        pub limits: crate::manifest::Limits,
        pub start: String,
        pub stages: HashMap<String, Stage>,
    }

    pub struct Stage {
        pub desc: String,
        pub system: String,
        pub schemas: Vec<Value>,
        pub tools: HashMap<String, Tool>,
        pub skills: HashMap<String, PathBuf>,
        pub next: Vec<String>,
        pub history: History,
    }

    #[derive(Clone)]
    pub struct Tool {
        pub run: Vec<String>,
        pub timeout: Duration,
        pub shell: bool,
        pub params: Option<Vec<Param>>,
        pub argv: Option<Vec<String>>,
        pub uid: Option<u32>,
        pub gid: Option<u32>,
    }

    #[derive(Default)]
    pub struct History {
        base: Vec<Entry>,
        replace: bool,
        turns: Option<usize>,
        percent: Option<usize>,
    }

    impl History {
        pub fn apply<'a>(&'a self, live: &'a [Entry]) -> Vec<&'a Entry> {
            let mut v: Vec<&Entry> = if self.replace {
                self.base.iter().collect()
            } else {
                self.base.iter().chain(live).collect()
            };
            if let Some(n) = self.turns
                && v.len() > n
            {
                v = v.split_off(v.len() - n);
            }
            if let Some(p) = self.percent {
                let cap = CONTEXT_CHARS * p / 100;
                let mut used = 0;
                let mut start = v.len();
                for e in v.iter().rev() {
                    let len = e.text().len();
                    if start < v.len() && used + len > cap {
                        break;
                    }
                    used += len;
                    start -= 1;
                }
                v = v.split_off(start);
            }
            v
        }
    }

    const ARGS_HINT: &str = "Command line arguments as a single string, like in a terminal. Quotes preserve spaces; * and variables are not expanded.";
    const SHELL_HINT: &str =
        "A shell command line. Pipes, globs and redirects are interpreted by the shell.";
    const ANSWER_RULE: &str = "Send the final answer as plain text.";
    const FACTS_RULE: &str = "Report only what tools actually returned; never invent output.";

    pub fn compile(path: &Path) -> anyhow::Result<Harness> {
        let file = std::fs::canonicalize(path).with_context(|| path.display().to_string())?;
        let text = std::fs::read_to_string(&file).with_context(|| file.display().to_string())?;
        let m: Manifest =
            serde_saphyr::from_str(&text).with_context(|| file.display().to_string())?;
        let dir = file.parent().unwrap_or(Path::new(".")).to_path_buf();
        build(m, dir).with_context(|| file.display().to_string())
    }

    fn build(m: Manifest, dir: PathBuf) -> anyhow::Result<Harness> {
        if m.model.url.trim().is_empty() {
            bail!("model.url is empty");
        }
        let prompts = uniq(&m.prompts, |p| p.name.as_str(), "prompts")?;
        let messages = uniq(&m.messages, |x| x.name.as_str(), "messages")?;
        let skills = uniq(&m.skills, |s| s.name.as_str(), "skills")?;
        let tools = uniq(&m.tools, |t| t.name.as_str(), "tools")?;
        let stage_index = uniq(&m.stages, |s| s.name.as_str(), "stages")?;
        for n in stage_index.keys() {
            if n.len() > 52 {
                bail!("stages: name \"{n}\" exceeds 52 chars (transfer_to_ prefix is reserved)");
            }
        }
        for p in &m.prompts {
            if p.policy != "pin" {
                bail!("prompts[{}].policy: only pin is supported", p.name);
            }
        }
        for n in tools.keys().chain(skills.keys()) {
            if n.starts_with("transfer_to_") || *n == "finish" {
                bail!("name \"{n}\" is reserved");
            }
        }
        for n in tools.keys() {
            if skills.contains_key(n) {
                bail!("name \"{n}\" is used by both a tool and a skill");
            }
        }

        let read = |rel: &str| {
            let p = dir.join(rel);
            std::fs::read_to_string(&p).with_context(|| p.display().to_string())
        };
        let mut prompt_text = HashMap::new();
        for (n, p) in &prompts {
            prompt_text.insert(*n, read(&p.path)?);
        }
        let mut message_entries = HashMap::new();
        for (n, x) in &messages {
            message_entries.insert(*n, load_entries(&dir.join(&x.path))?);
        }
        let mut skill_paths = HashMap::new();
        for (n, s) in &skills {
            let p = dir.join(&s.instr);
            if !p.is_file() {
                bail!("skills[{n}].instr: file not found {}", p.display());
            }
            skill_paths.insert(*n, p);
        }
        let mut tool_defs = HashMap::new();
        for (n, t) in &tools {
            let run: Vec<String> = t.run.split_whitespace().map(str::to_owned).collect();
            if run.is_empty() {
                bail!("tools[{n}].run: empty command");
            }
            if t.shell && t.schema.is_some() {
                bail!("tools[{n}]: shell and schema are mutually exclusive");
            }
            let (params, argv) = match &t.schema {
                Some(rel) => {
                    let s = load_tool_schema(&dir.join(rel), n)?;
                    (Some(s.params), s.argv)
                }
                None => (None, None),
            };
            tool_defs.insert(
                (*n).to_owned(),
                Tool {
                    run,
                    timeout: t.timeout,
                    shell: t.shell,
                    params,
                    argv,
                    uid: t.uid,
                    gid: t.gid,
                },
            );
        }
        for sd in &m.stages {
            refs(&sd.prompt, &prompts, &format!("stages[{}].prompt", sd.name))?;
            refs(&sd.tools, &tools, &format!("stages[{}].tools", sd.name))?;
            refs(&sd.skills, &skills, &format!("stages[{}].skills", sd.name))?;
            refs(&sd.next, &stage_index, &format!("stages[{}].next", sd.name))?;
        }

        let synth;
        let list: &[StageDef] = if m.stages.is_empty() {
            synth = StageDef {
                name: "default".to_string(),
                desc: "Single stage.".to_string(),
                prompt: m.prompts.iter().map(|p| p.name.clone()).collect(),
                tools: m.tools.iter().map(|t| t.name.clone()).collect(),
                skills: m.skills.iter().map(|s| s.name.clone()).collect(),
                next: Vec::new(),
                history: None,
            };
            std::slice::from_ref(&synth)
        } else {
            &m.stages
        };

        let mut stages = HashMap::new();
        for sd in list {
            let mut parts: Vec<String> = sd
                .prompt
                .iter()
                .map(|n| prompt_text[n.as_str()].clone())
                .collect();
            parts.push(format!("Current stage: {}. {}", sd.name, sd.desc));
            parts.push(ANSWER_RULE.to_string());
            if !sd.tools.is_empty() {
                parts.push(FACTS_RULE.to_string());
            }
            let mut schemas = Vec::new();
            for n in &sd.tools {
                let def = tools[n.as_str()];
                let desc = format!("{}\nCommand: {}.", def.desc, def.run);
                schemas.push(schema(n, &desc, tool_params(&tool_defs[n.as_str()])));
            }
            for n in &sd.skills {
                schemas.push(schema(
                    n,
                    &skills[n.as_str()].desc,
                    json!({"type": "object", "properties": {}}),
                ));
            }
            for n in &sd.next {
                schemas.push(schema(
                    &format!("transfer_to_{n}"),
                    &stage_index[n.as_str()].desc,
                    json!({"type": "object", "properties": {}}),
                ));
            }

            let history = resolve_history(&sd.history, &message_entries, &dir, &sd.name)?;
            let stage = Stage {
                desc: sd.desc.clone(),
                system: parts.join("\n\n"),
                schemas,
                tools: sd
                    .tools
                    .iter()
                    .map(|n| (n.clone(), tool_defs[n.as_str()].clone()))
                    .collect(),
                skills: sd
                    .skills
                    .iter()
                    .map(|n| (n.clone(), skill_paths[n.as_str()].clone()))
                    .collect(),
                next: sd.next.clone(),
                history,
            };
            stages.insert(sd.name.clone(), stage);
        }

        Ok(Harness {
            dir,
            endpoint: format!("{}/chat/completions", m.model.url.trim_end_matches('/')),
            model: m.model.name,
            key: std::env::var("UZDA_API_KEY").unwrap_or_default(),
            limits: m.limits,
            start: list[0].name.clone(),
            stages,
        })
    }

    fn schema(name: &str, desc: &str, params: Value) -> Value {
        json!({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    }

    fn tool_params(t: &Tool) -> Value {
        let Some(params) = &t.params else {
            let hint = if t.shell { SHELL_HINT } else { ARGS_HINT };
            return json!({
                "type": "object",
                "properties": {"args": {"type": "string", "description": hint}},
                "additionalProperties": false
            });
        };
        let mut props = Map::new();
        let mut required = Vec::new();
        for p in params {
            let mut spec = json!({"type": p.kind.label()});
            if !p.desc.is_empty() {
                spec["description"] = json!(p.desc);
            }
            if let Some(d) = &p.default {
                spec["default"] = d.clone();
            }
            props.insert(p.name.clone(), spec);
            if p.required && p.default.is_none() {
                required.push(p.name.clone());
            }
        }
        json!({"type": "object", "properties": props, "required": required, "additionalProperties": false})
    }

    fn load_tool_schema(p: &Path, tool: &str) -> anyhow::Result<ToolSchema> {
        let at = || format!("tools[{tool}].schema: {}", p.display());
        let text = std::fs::read_to_string(p).with_context(at)?;
        let s: ToolSchema = serde_saphyr::from_str(&text).with_context(at)?;
        for (i, param) in s.params.iter().enumerate() {
            if !valid_name(&param.name) {
                bail!(
                    "tools[{tool}].schema: params[{}].name: ascii letters, digits, _ or - only",
                    param.name
                );
            }
            if s.params[..i].iter().any(|p| p.name == param.name) {
                bail!("tools[{tool}].schema: duplicate param \"{}\"", param.name);
            }
            if let Some(d) = &param.default
                && !param.kind.matches(d)
            {
                bail!(
                    "tools[{tool}].schema: params[{}].default: expected {}",
                    param.name,
                    param.kind.label()
                );
            }
        }
        if let Some(argv) = &s.argv {
            for t in argv {
                for ph in placeholders(t) {
                    if !s.params.iter().any(|p| p.name == ph) {
                        bail!("tools[{tool}].schema: argv: unknown param {{{ph}}}");
                    }
                }
            }
        }
        Ok(s)
    }

    fn placeholders(t: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut rest = t;
        while let Some(i) = rest.find('{') {
            rest = &rest[i + 1..];
            if let Some(j) = rest.find('}') {
                out.push(&rest[..j]);
                rest = &rest[j + 1..];
            }
        }
        out
    }

    fn resolve_history(
        spec: &Option<HistorySpec>,
        messages: &HashMap<&str, Vec<Entry>>,
        dir: &Path,
        stage: &str,
    ) -> anyhow::Result<History> {
        let field = format!("stages[{stage}].history");
        let mut h = History::default();
        let Some(spec) = spec else { return Ok(h) };
        let (source, mode, turns, percent) = match spec {
            HistorySpec::Alias(a) => match a.as_str() {
                "full" => (None, None, None, None),
                "brief" => (None, None, Some(10), None),
                _ => bail!("{field}: unknown mode \"{a}\""),
            },
            HistorySpec::Policy {
                r#use,
                mode,
                turns,
                percent,
            } => (r#use.as_deref(), mode.as_deref(), *turns, *percent),
        };
        if let Some(p) = percent
            && !(1..=100).contains(&p)
        {
            bail!("{field}.percent: must be 1..100");
        }
        match mode.unwrap_or("overlay") {
            "overlay" => {}
            "replace" => h.replace = true,
            other => bail!("{field}.mode: unknown mode \"{other}\""),
        }
        if h.replace && source.is_none() {
            bail!("{field}.mode: replace requires use");
        }
        h.turns = turns;
        h.percent = percent;
        if let Some(u) = source {
            h.base = match messages.get(u) {
                Some(entries) => entries.clone(),
                None => load_entries(&dir.join(u)).with_context(|| format!("{field}.use"))?,
            };
            if h.replace && h.base.is_empty() {
                bail!("{field}.use: empty history for replace");
            }
        }
        Ok(h)
    }

    fn load_entries(p: &Path) -> anyhow::Result<Vec<Entry>> {
        let text = std::fs::read_to_string(p).with_context(|| p.display().to_string())?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        serde_saphyr::from_str(&text).with_context(|| p.display().to_string())
    }
}

// ===== argv =====
mod argv {
    use serde_json::{Map, Value};
    use std::str::Chars;

    pub fn val_str(v: &Value) -> String {
        v.as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| v.to_string())
    }

    pub fn args_line(args: &Map<String, Value>) -> Result<String, String> {
        let v = if let Some(v) = args.get("args") {
            Some(v)
        } else if args.len() == 1 {
            args.values().next()
        } else if args.is_empty() {
            None
        } else {
            return Err("expected args: a single argument string".to_string());
        };
        match v {
            None => Ok(String::new()),
            Some(Value::String(s)) => Ok(s.clone()),
            Some(v @ (Value::Number(_) | Value::Bool(_))) => Ok(v.to_string()),
            Some(_) => Err("args: expected a string".to_string()),
        }
    }

    fn quoted(it: &mut Chars<'_>, delim: char) -> Result<String, String> {
        let mut out = String::new();
        while let Some(c) = it.next() {
            if c == delim {
                return Ok(out);
            }
            if c == '\\' && delim == '"' {
                match it.next() {
                    Some(e @ ('"' | '\\')) => out.push(e),
                    Some(e) => {
                        out.push('\\');
                        out.push(e);
                    }
                    None => break,
                }
            } else {
                out.push(c);
            }
        }
        Err(format!("unterminated {delim} quote in args"))
    }

    pub fn split_args(s: &str) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut has = false;
        let mut it = s.chars();
        while let Some(c) = it.next() {
            match c {
                ' ' | '\t' | '\n' | '\r' => {
                    if has {
                        out.push(std::mem::take(&mut cur));
                        has = false;
                    }
                }
                '\'' | '"' => {
                    has = true;
                    cur.push_str(&quoted(&mut it, c)?);
                }
                '\\' => {
                    has = true;
                    cur.push(
                        it.next()
                            .ok_or_else(|| "trailing backslash in args".to_string())?,
                    );
                }
                c => {
                    has = true;
                    cur.push(c);
                }
            }
        }
        if has {
            out.push(cur);
        }
        Ok(out)
    }
}

// ===== exec =====
mod exec {
    use crate::budget::MAX_OUTPUT;
    use crate::harness::{Harness, Tool};
    use crate::model::ToolCall;
    use serde_json::{Map, Value};
    use std::path::Path;
    use std::process::Stdio;
    use std::sync::Arc;
    use tokio::io::AsyncReadExt;
    use tokio::process::Command;
    use tokio::sync::Semaphore;

    #[derive(Clone)]
    pub struct Call {
        pub name: String,
        pub args: Map<String, Value>,
        pub err: Option<String>,
    }

    impl Call {
        pub fn parse(tc: ToolCall) -> Call {
            let raw = tc.function.arguments.trim();
            let raw = if raw.is_empty() { "{}" } else { raw };
            let (args, err) = match serde_json::from_str::<Value>(raw) {
                Ok(Value::Object(m)) => (m, None),
                Ok(_) => (Map::new(), Some("arguments: expected an object".to_owned())),
                Err(e) => (Map::new(), Some(format!("arguments: {e}"))),
            };
            Call {
                name: tc.function.name,
                args,
                err,
            }
        }
    }

    pub fn fmt_call(name: &str, args: &Map<String, Value>) -> String {
        let inner: Vec<String> = args
            .iter()
            .map(|(k, v)| format!("{k}={}", crate::yaml::quote(&crate::argv::val_str(v))))
            .collect();
        format!("{name}({})", inner.join(", "))
    }

    pub struct Outcome {
        pub text: String,
        pub ok: bool,
        pub effect: Effect,
    }

    pub enum Effect {
        None,
        Stage(String),
        Skill(String, String),
    }

    impl Outcome {
        pub fn err(text: String) -> Outcome {
            Outcome {
                text,
                ok: false,
                effect: Effect::None,
            }
        }

        pub fn ok(text: String, effect: Effect) -> Outcome {
            Outcome {
                text,
                ok: true,
                effect,
            }
        }
    }

    pub struct Output {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        code: Option<i32>,
        ok: bool,
        cut: bool,
    }

    impl Output {
        fn render(self) -> String {
            let out = String::from_utf8_lossy(&self.stdout);
            let err = String::from_utf8_lossy(&self.stderr);
            let mut text = if self.ok && !self.cut {
                out.into_owned()
            } else {
                let mut body = err.trim().to_owned();
                let tail = out.trim();
                if !tail.is_empty() {
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str(tail);
                }
                if body.is_empty() {
                    body = "(empty)".to_string();
                }
                match self.code {
                    Some(code) => format!("exit {code}: {body}"),
                    None => format!("killed: {body}"),
                }
            };
            if self.cut {
                text.push_str("\n...[output limit reached, process killed]");
            }
            if text.is_empty() {
                text = "(empty)".to_string();
            }
            if text.len() > MAX_OUTPUT {
                let mut end = MAX_OUTPUT;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
            }
            text
        }
    }

    async fn drain<R: tokio::io::AsyncRead + Unpin>(src: Option<R>, cap: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        if let Some(r) = src {
            let _ = r.take(cap).read_to_end(&mut buf).await;
        }
        buf
    }

    async fn capture(mut cmd: Command, limit: std::time::Duration) -> Result<Output, String> {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| format!("spawn: {e}"))?;
        let out = child.stdout.take();
        let err = child.stderr.take();
        let cap = (MAX_OUTPUT + 1) as u64;
        let job = async move {
            let (stdout, stderr) = tokio::join!(drain(out, cap), drain(err, cap));
            let cut = stdout.len() > MAX_OUTPUT || stderr.len() > MAX_OUTPUT;
            if cut {
                let _ = child.start_kill();
            }
            let status = child.wait().await.map_err(|e| format!("wait: {e}"))?;
            Ok::<_, String>(Output {
                stdout,
                stderr,
                code: status.code(),
                ok: status.success(),
                cut,
            })
        };
        match tokio::time::timeout(limit, job).await {
            Ok(res) => res,
            Err(_) => Err(format!("timeout {}s", limit.as_secs())),
        }
    }

    fn apply_ids(cmd: &mut Command, uid: Option<u32>, gid: Option<u32>) {
        #[cfg(unix)]
        {
            if let Some(g) = gid {
                cmd.gid(g);
            }
            if let Some(u) = uid {
                cmd.uid(u);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (cmd, uid, gid);
        }
    }

    async fn run_tool(dir: &Path, t: &Tool, call: &Call) -> Outcome {
        let argv = match t.argv(&call.args) {
            Ok(argv) => argv,
            Err(e) => return Outcome::err(e),
        };
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]).current_dir(dir);
        apply_ids(&mut cmd, t.uid, t.gid);
        match capture(cmd, t.timeout).await {
            Ok(out) => {
                let ok = out.ok && !out.cut;
                Outcome {
                    text: out.render(),
                    ok,
                    effect: Effect::None,
                }
            }
            Err(e) => Outcome::err(e),
        }
    }

    pub async fn execute(
        tool_gate: Arc<Semaphore>,
        h: Arc<Harness>,
        stage: String,
        call: Call,
    ) -> Outcome {
        if let Some(e) = call.err {
            return Outcome::err(e);
        }
        let Some(st) = h.stages.get(&stage) else {
            return Outcome::err(format!("unknown stage: {stage}"));
        };
        if let Some(t) = st.tools.get(&call.name) {
            let Ok(_permit) = tool_gate.acquire().await else {
                return Outcome::err("tool queue closed".to_string());
            };
            return run_tool(&h.dir, t, &call).await;
        }
        if let Some(instr) = st.skills.get(&call.name) {
            if !call.args.is_empty() {
                return Outcome::err("skill does not accept arguments".to_string());
            }
            return match tokio::fs::read_to_string(instr).await {
                Ok(text) if text.trim().is_empty() => {
                    Outcome::err("skill instruction is empty".to_string())
                }
                Ok(text) if text.len() > 1024 * 1024 => {
                    Outcome::err("skill instruction exceeds 1 MiB".to_string())
                }
                Ok(text) => Outcome::ok(
                    format!("skill activated: {}", call.name),
                    Effect::Skill(call.name.clone(), text),
                ),
                Err(e) => Outcome::err(format!("instruction: {e}")),
            };
        }
        if let Some(target) = call.name.strip_prefix("transfer_to_") {
            return if st.next.iter().any(|n| n == target) {
                Outcome::ok(
                    "stage changed".to_string(),
                    Effect::Stage(target.to_owned()),
                )
            } else {
                Outcome::err(format!("transition not allowed: {target}"))
            };
        }
        Outcome::err(format!("unknown call: {}", call.name))
    }
}

// ===== model =====
mod model {
    use crate::harness::Harness;
    use serde::Deserialize;
    use serde_json::{Value, json};

    fn null_vec<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
    where
        D: serde::Deserializer<'de>,
        T: Deserialize<'de>,
    {
        Ok(Option::<Vec<T>>::deserialize(d)?.unwrap_or_default())
    }

    #[derive(Deserialize)]
    pub struct Message {
        #[serde(default)]
        pub content: Option<String>,
        #[serde(default, deserialize_with = "null_vec")]
        pub tool_calls: Vec<ToolCall>,
    }

    #[derive(Deserialize)]
    pub struct ToolCall {
        pub function: Function,
    }

    #[derive(Deserialize)]
    pub struct Function {
        pub name: String,
        #[serde(default)]
        pub arguments: String,
    }

    pub async fn call_model(
        http: &reqwest::Client,
        h: &Harness,
        messages: Vec<Value>,
        schemas: &[Value],
        mut on_delta: impl FnMut(&str),
    ) -> Result<Message, String> {
        let mut resp = http
            .post(&h.endpoint)
            .bearer_auth(&h.key)
            .json(&json!({"model": h.model, "messages": messages, "tools": schemas, "tool_choice": if schemas.is_empty() { "none" } else { "auto" }, "stream": true}))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.map_err(|e| e.to_string())?;
            return Err(format!(
                "{status}: {}",
                body.chars().take(500).collect::<String>()
            ));
        }
        let mut pending = Vec::new();
        let mut content = String::new();
        let mut calls: std::collections::BTreeMap<u64, Function> =
            std::collections::BTreeMap::new();
        let mut done = false;
        while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
            pending.extend_from_slice(&chunk);
            if pending.len() > 1024 * 1024 || content.len() > 1024 * 1024 {
                return Err("model response exceeds limit".into());
            }
            while let Some(pos) = pending.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = pending.drain(..=pos).collect();
                let line = std::str::from_utf8(&line)
                    .map_err(|e| format!("stream UTF-8: {e}"))?
                    .trim();
                let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if data == "[DONE]" {
                    done = true;
                    continue;
                }
                let event: Value =
                    serde_json::from_str(data).map_err(|e| format!("stream parse: {e}"))?;
                if let Some(error) = event.get("error") {
                    return Err(format!("model stream: {error}"));
                }
                let Some(delta) = event
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|v| v.first())
                    .and_then(|c| c.get("delta"))
                else {
                    continue;
                };
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    content.push_str(text);
                    on_delta(text);
                }
                if let Some(parts) = delta.get("tool_calls").and_then(Value::as_array) {
                    for part in parts {
                        let index = part
                            .get("index")
                            .and_then(Value::as_u64)
                            .ok_or("tool call missing index")?;
                        if index >= crate::budget::MAX_CALLS as u64 {
                            return Err("too many tool calls".into());
                        }
                        let function = calls.entry(index).or_insert_with(|| Function {
                            name: String::new(),
                            arguments: String::new(),
                        });
                        if let Some(f) = part.get("function") {
                            if let Some(s) = f.get("name").and_then(Value::as_str) {
                                function.name.push_str(s);
                            }
                            if let Some(s) = f.get("arguments").and_then(Value::as_str) {
                                function.arguments.push_str(s);
                            }
                            if function.arguments.len() > 1024 * 1024 || function.name.len() > 1024
                            {
                                return Err("tool call exceeds limit".into());
                            }
                        }
                    }
                }
            }
        }
        if !done {
            return Err("model stream ended before [DONE]".into());
        }
        Ok(Message {
            content: Some(content),
            tool_calls: calls
                .into_values()
                .map(|function| ToolCall { function })
                .collect(),
        })
    }
}

// ===== session =====
mod session {
    use crate::budget::{MAX_CALLS, MODEL_TIMEOUT};
    use crate::exec::{self, Call, Effect};
    use crate::harness::{Harness, Stage};
    use crate::model::{self, Message};
    use crate::trace::{Entry, Trace, UserText};
    use serde_json::{Value, json};
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::sync::{Semaphore, broadcast, mpsc, watch};

    pub struct Inbox {
        pub from: Option<String>,
        pub text: String,
    }

    #[derive(Clone)]
    pub struct Frame {
        pub seq: u64,
        pub kind: String,
        pub body: String,
    }

    impl Frame {
        pub fn reload() -> Frame {
            Frame {
                seq: 0,
                kind: "lagged".to_string(),
                body: "{\"reload\":true}".to_string(),
            }
        }
    }

    pub struct Channel {
        tx: broadcast::Sender<Frame>,
        replay: Mutex<VecDeque<Frame>>,
    }

    impl Channel {
        pub fn new() -> Arc<Channel> {
            let (tx, _) = broadcast::channel(crate::budget::REPLAY);
            Arc::new(Channel {
                tx,
                replay: Mutex::new(VecDeque::new()),
            })
        }

        pub fn push(&self, frame: Frame) {
            let mut q = crate::lock(&self.replay);
            if q.len() == crate::budget::REPLAY {
                q.pop_front();
            }
            q.push_back(frame.clone());
            let _ = self.tx.send(frame);
        }

        pub fn subscribe(&self) -> broadcast::Receiver<Frame> {
            self.tx.subscribe()
        }

        pub fn snapshot(&self, since: u64) -> Vec<Frame> {
            let q = crate::lock(&self.replay);
            let gap = since > 0 && q.front().is_some_and(|f| since < f.seq);
            let mut out = if gap {
                vec![Frame::reload()]
            } else {
                Vec::new()
            };
            out.extend(q.iter().filter(|f| f.seq > since).cloned());
            out
        }
    }

    pub struct Stop(watch::Sender<bool>);

    impl Stop {
        pub fn cancel(&self) {
            let _ = self.0.send(true);
        }
    }

    #[derive(Clone)]
    pub struct Stopped(watch::Receiver<bool>);

    impl Stopped {
        pub fn is_set(&self) -> bool {
            *self.0.borrow()
        }

        pub async fn wait(&mut self) {
            let _ = self.0.wait_for(|s| *s).await;
        }
    }

    fn stop_pair() -> (Stop, Stopped) {
        let (tx, rx) = watch::channel(false);
        (Stop(tx), Stopped(rx))
    }

    pub struct Handle {
        pub inbox: mpsc::UnboundedSender<Inbox>,
        pub stop: Stop,
        pub channel: Arc<Channel>,
    }

    pub type Registry = Arc<Mutex<HashMap<String, Handle>>>;

    struct LiveGuard {
        live: Registry,
        id: String,
    }

    impl Drop for LiveGuard {
        fn drop(&mut self) {
            crate::lock(&self.live).remove(&self.id);
        }
    }

    #[derive(Clone, Copy)]
    pub enum End {
        Answer,
        Finish,
        TurnLimit(u64),
        TimeLimit(u64),
        Stopped,
        ModelDown,
        NoAnswer,
        TraceError,
    }

    impl End {
        pub fn tag(self) -> &'static str {
            match self {
                Self::Answer => "answer",
                Self::Finish => "finish",
                Self::TurnLimit(_) => "turn_limit",
                Self::TimeLimit(_) => "time_limit",
                Self::Stopped => "stopped",
                Self::ModelDown => "model_down",
                Self::NoAnswer => "no_answer",
                Self::TraceError => "trace_error",
            }
        }

        pub fn message(self) -> String {
            match self {
                Self::TurnLimit(n) => format!("turn limit ({n})"),
                Self::TimeLimit(s) => format!("time limit ({s}s)"),
                other => other.tag().replace('_', " "),
            }
        }
    }

    pub struct Rig {
        pub http: reqwest::Client,
        pub model_gate: Arc<Semaphore>,
        pub tool_gate: Arc<Semaphore>,
        pub live: Registry,
    }

    pub struct Session {
        rig: Rig,
        harness: Arc<Harness>,
        trace: Trace,
        pub channel: Arc<Channel>,
        stage: String,
        turns: u64,
        deadline: Option<Instant>,
        skill: Option<(String, String)>,
        hitch: Option<String>,
        query: String,
        seq: u64,
        inbox: mpsc::UnboundedReceiver<Inbox>,
        stop: Stopped,
        stalls: u32,
        errors: u32,
        _guard: LiveGuard,
    }

    impl Session {
        pub fn new(
            rig: Rig,
            harness: Arc<Harness>,
            trace: Trace,
            query: String,
            hitch: Option<String>,
        ) -> crate::fail::Result<Session> {
            let channel = Channel::new();
            let (inbox_tx, inbox) = mpsc::unbounded_channel();
            let (stop, stopped) = stop_pair();
            let stage = trace.stage(&harness);
            let deadline = harness
                .limits
                .time
                .and_then(|s| Instant::now().checked_add(Duration::from_secs(s)));
            {
                let mut live = crate::lock(&rig.live);
                match live.entry(trace.id.clone()) {
                    std::collections::hash_map::Entry::Occupied(_) => {
                        return Err(crate::fail::Error::Conflict);
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(Handle {
                            inbox: inbox_tx,
                            stop,
                            channel: channel.clone(),
                        });
                    }
                }
            }
            let guard = LiveGuard {
                live: rig.live.clone(),
                id: trace.id.clone(),
            };
            Ok(Session {
                rig,
                harness,
                trace,
                channel,
                stage,
                turns: 0,
                deadline,
                skill: None,
                hitch,
                query,
                seq: 0,
                inbox,
                stop: stopped,
                stalls: 0,
                errors: 0,
                _guard: guard,
            })
        }

        fn emit(&mut self, kind: &str, mut body: Value) {
            if matches!(
                kind,
                "session_start" | "stage_enter" | "done" | "error" | "queued" | "call" | "result"
            ) {
                eprintln!(
                    "[{}] {} {}",
                    self.trace.id,
                    kind,
                    match kind {
                        "call" => format!("name={}", body["name"]),
                        "result" => format!("name={} success={}", body["name"], body["success"]),
                        _ => body.to_string(),
                    }
                );
            }
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            self.seq = self.seq.saturating_add(1).max(now);
            body["seq"] = json!(self.seq);
            self.channel.push(Frame {
                seq: self.seq,
                kind: kind.to_string(),
                body: body.to_string(),
            });
        }

        async fn record(&mut self, e: Entry) {
            if self.trace.broken {
                return;
            }
            if let Err(err) = self.trace.push(e).await {
                self.trace.broken = true;
                self.emit("error", json!({"message": format!("trace: {err}")}));
            }
        }

        fn user_entry(&self, from: Option<String>, text: String) -> Entry {
            match &self.hitch {
                Some(h) => Entry::User(UserText::From {
                    from: from.unwrap_or_else(|| h.clone()),
                    text,
                }),
                None => Entry::User(UserText::Plain(text)),
            }
        }

        fn exhausted(&self) -> Option<End> {
            if self.trace.broken {
                return Some(End::TraceError);
            }
            if self.stop.is_set() {
                return Some(End::Stopped);
            }
            if self.turns >= self.harness.limits.turns {
                return Some(End::TurnLimit(self.harness.limits.turns));
            }
            if self.deadline.is_some_and(|d| Instant::now() >= d) {
                return Some(End::TimeLimit(self.harness.limits.time.unwrap_or_default()));
            }
            None
        }

        async fn fail(&mut self, message: String) {
            self.emit("error", json!({"message": message}));
        }

        async fn drain_inbox(&mut self) {
            while let Ok(m) = self.inbox.try_recv() {
                let e = self.user_entry(m.from, m.text);
                self.record(e).await;
            }
        }

        pub async fn run(mut self) {
            self.emit("session_start", json!({"trace_id": self.trace.id}));

            let query = std::mem::take(&mut self.query);
            let user = self.user_entry(None, query);
            self.record(user).await;
            let end = match self.exhausted() {
                Some(end) => end,
                None => {
                    let desc = self.harness.stages[&self.stage].desc.clone();
                    self.emit("stage_enter", json!({"stage": self.stage, "desc": desc}));
                    self.cycle().await
                }
            };
            self.emit(
                "done",
                json!({"turns": self.turns, "reason": end.tag(), "message": end.message()}),
            );
        }

        async fn cycle(&mut self) -> End {
            let h = self.harness.clone();
            loop {
                if let Some(end) = self.exhausted() {
                    return end;
                }
                self.drain_inbox().await;
                let st = &h.stages[&self.stage];
                let planning = self.skill.is_some();
                let messages = self.context(st);
                let limit = self
                    .deadline
                    .map(|d| d.saturating_duration_since(Instant::now()))
                    .unwrap_or(MODEL_TIMEOUT);
                self.turns += 1;
                let schemas = if planning { &[][..] } else { &st.schemas };
                let message = match self.ask(&h, messages, schemas, limit, planning).await {
                    Ok(m) => {
                        self.errors = 0;
                        m
                    }
                    Err(e) => {
                        if e == "stopped" || self.stop.is_set() {
                            return End::Stopped;
                        }
                        self.errors += 1;
                        eprintln!(
                            "[{}] model attempt {} failed: {}",
                            self.trace.id, self.errors, e
                        );
                        if e.starts_with("partial response:") || self.errors >= 3 {
                            self.fail(format!("model error: {e}")).await;
                            return End::ModelDown;
                        }
                        tokio::time::sleep(Duration::from_millis(400 * self.errors as u64)).await;
                        continue;
                    }
                };
                if planning {
                    let plan = message.content.unwrap_or_default();
                    if !message.tool_calls.is_empty() || plan.trim().is_empty() {
                        self.stalls += 1;
                        self.fail(
                            "skill planning requires a non-empty text response without tool calls"
                                .to_string(),
                        )
                        .await;
                        if self.stalls >= 2 {
                            return End::NoAnswer;
                        }
                        continue;
                    }
                    self.record(Entry::Model(format!("Plan:\n{plan}"))).await;
                    if self.trace.broken {
                        return End::TraceError;
                    }
                    self.skill = None;
                    self.stalls = 0;
                    continue;
                }
                if let Some(end) = self.handle_reply(message).await {
                    return end;
                }
            }
        }

        async fn handle_reply(&mut self, message: Message) -> Option<End> {
            let text = message.content.unwrap_or_default();
            let has_text = !text.trim().is_empty();
            if message.tool_calls.is_empty() {
                if !has_text {
                    self.stalls += 1;
                    self.fail("empty model reply".to_string()).await;
                    return if self.stalls >= 2 {
                        Some(End::NoAnswer)
                    } else {
                        None
                    };
                }
                self.record(Entry::Model(text.clone())).await;
                self.emit("answer", json!({"content": text}));
                return Some(End::Answer);
            }
            self.stalls = 0;
            let mut calls: Vec<Call> = message.tool_calls.into_iter().map(Call::parse).collect();
            let skills = &self.harness.stages[&self.stage].skills;
            if calls.len() > 1 && calls.iter().any(|c| skills.contains_key(&c.name)) {
                for c in &calls {
                    self.record(Entry::Call(exec::fmt_call(&c.name, &c.args)))
                        .await;
                    self.record(Entry::Result(
                        "skill must be called alone; no calls were executed".to_string(),
                    ))
                    .await;
                }
                self.fail("skill must be called alone; no calls were executed".to_string())
                    .await;
                return None;
            }
            let finish_at = calls.iter().position(|c| c.name == "finish");
            if has_text {
                self.record(Entry::Model(text.clone())).await;
                if finish_at.is_some() {
                    self.emit("answer", json!({"content": text}));
                } else {
                    self.emit("thought", json!({"text": text}));
                }
            }
            if let Some(i) = finish_at {
                if !has_text {
                    self.fail(
                        "finish requires a text answer; asking the model to continue".to_string(),
                    )
                    .await;
                    self.record(Entry::Result(
                        "Provide the final answer as text; do not call finish without an answer."
                            .to_string(),
                    ))
                    .await;
                    return None;
                }
                let f = &calls[i];
                let reason: String = f
                    .args
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("done")
                    .chars()
                    .take(200)
                    .collect();
                let mut args = f.args.clone();
                args.insert("reason".to_string(), Value::String(reason.clone()));
                self.record(Entry::Call(exec::fmt_call(&f.name, &args)))
                    .await;
                self.record(Entry::Result(reason)).await;
                return Some(End::Finish);
            }
            if calls.len() > MAX_CALLS {
                self.fail(format!(
                    "too many calls: {} (limit {MAX_CALLS})",
                    calls.len()
                ))
                .await;
                calls.truncate(MAX_CALLS);
            }
            self.run_calls(calls).await;
            None
        }

        async fn run_calls(&mut self, calls: Vec<Call>) {
            let h = self.harness.clone();
            let mut jobs = Vec::with_capacity(calls.len());
            for c in &calls {
                self.record(Entry::Call(exec::fmt_call(&c.name, &c.args)))
                    .await;
                self.emit("call", json!({"name": c.name, "args": c.args}));
                jobs.push(tokio::spawn(exec::execute(
                    self.rig.tool_gate.clone(),
                    h.clone(),
                    self.stage.clone(),
                    c.clone(),
                )));
            }
            let mut next_stage = None;
            let mut next_skill = None;
            for (c, job) in calls.iter().zip(jobs) {
                let out = job
                    .await
                    .unwrap_or_else(|e| exec::Outcome::err(format!("execution: {e}")));
                self.record(Entry::Result(out.text.clone())).await;
                self.emit(
                    "result",
                    json!({"name": c.name, "output": out.text, "success": out.ok}),
                );
                match out.effect {
                    Effect::Stage(s) => next_stage = Some(s),
                    Effect::Skill(n, t) => next_skill = Some((n, t)),
                    Effect::None => {}
                }
            }
            if let Some(s) = next_stage {
                self.record(Entry::Stage(s.clone())).await;
                if self.trace.broken {
                    return;
                }
                self.stage = s;
                let desc = self.harness.stages[&self.stage].desc.clone();
                self.emit("stage_enter", json!({"stage": self.stage, "desc": desc}));
            }
            if let Some(sk) = next_skill {
                self.skill = Some(sk);
            }
        }

        async fn ask(
            &mut self,
            h: &Harness,
            messages: Vec<Value>,
            schemas: &[Value],
            limit: Duration,
            planning: bool,
        ) -> Result<Message, String> {
            let gate = self.rig.model_gate.clone();
            let mut stopped = self.stop.clone();
            let permit = match gate.try_acquire() {
                Ok(p) => p,
                Err(_) => {
                    self.emit("queued", json!({"turn": self.turns}));
                    tokio::select! {
                        p = gate.acquire() => p.map_err(|_| "queue closed".to_owned())?,
                        _ = stopped.wait() => return Err("stopped".to_string()),
                    }
                }
            };
            if self.stop.is_set() {
                return Err("stopped".to_string());
            }
            let http = self.rig.http.clone();
            let mut streamed = false;
            let call = model::call_model(&http, h, messages, schemas, |text| {
                if !planning {
                    streamed |= !text.is_empty();
                    self.emit("delta", json!({"content": text}));
                }
            });
            let out = tokio::select! {
                r = tokio::time::timeout(limit, call) => match r {
                    Ok(inner) => inner,
                    Err(_) => Err("timeout".to_string()),
                },
                _ = stopped.wait() => Err("stopped".to_string()),
            };
            drop(permit);
            out.map_err(|e| {
                if streamed {
                    format!("partial response: {e}")
                } else {
                    e
                }
            })
        }

        fn preprocess(messages: Vec<Value>) -> Vec<Value> {
            messages
        }

        fn context(&mut self, st: &Stage) -> Vec<Value> {
            let mut system = st.system.clone();
            if let Some((name, instr)) = &self.skill {
                system.push_str(&format!("\n\nSkill {name} (planning turn only):\n{instr}\n\nWrite a concise actionable plan for yourself based on this skill and the current task. Do not answer the user or call tools in this turn."));
            }
            let mut messages = vec![json!({"role": "system", "content": system})];
            let mut role = "";
            let mut buf = String::new();
            for e in st.history.apply(&self.trace.entries) {
                let (r, t) = match *e {
                    Entry::User(_) => ("user", e.text()),
                    Entry::Model(_) | Entry::Call(_) => ("assistant", e.text()),
                    Entry::Result(_) => ("user", e.text()),
                    Entry::System(_) | Entry::Stage(_) => continue,
                };
                if r != role {
                    if !buf.is_empty() {
                        messages.push(json!({"role": role, "content": std::mem::take(&mut buf)}));
                    }
                    role = r;
                }
                if !buf.is_empty() {
                    buf.push_str("\n\n");
                }
                buf.push_str(t);
            }
            if !buf.is_empty() {
                messages.push(json!({"role": role, "content": buf}));
            }
            if self.stalls == 1 {
                messages.push(json!({
                    "role": "user",
                    "content": "Your previous reply was empty. Answer the participant with text or use an available tool."
                }));
            }
            Self::preprocess(messages)
        }
    }

    impl crate::harness::Tool {
        pub fn argv(&self, args: &serde_json::Map<String, Value>) -> Result<Vec<String>, String> {
            let mut argv = self.run.clone();
            if self.shell {
                let line = crate::argv::args_line(args)?;
                if line.trim().is_empty() {
                    return Err("shell: empty command line".to_string());
                }
                argv.push("-c".to_string());
                argv.push(line);
                return Ok(argv);
            }
            match &self.params {
                None => {
                    let line = crate::argv::args_line(args)?;
                    if !line.is_empty() {
                        argv.extend(crate::argv::split_args(&line)?);
                    }
                }
                Some(params) => {
                    for k in args.keys() {
                        if !params.iter().any(|p| &p.name == k) {
                            return Err(format!("unexpected argument \"{k}\""));
                        }
                    }
                    let mut vals = HashMap::new();
                    for p in params {
                        let Some(v) = args.get(&p.name).or(p.default.as_ref()) else {
                            if p.required {
                                return Err(format!("missing required argument \"{}\"", p.name));
                            }
                            continue;
                        };
                        if !p.kind.matches(v) {
                            return Err(format!(
                                "argument \"{}\": expected {}",
                                p.name,
                                p.kind.label()
                            ));
                        }
                        vals.insert(p.name.clone(), crate::argv::val_str(v));
                    }
                    match &self.argv {
                        Some(tmpl) => argv.extend(expand(tmpl, params, &vals)),
                        None => {
                            argv.extend(params.iter().filter_map(|p| vals.get(&p.name).cloned()))
                        }
                    }
                }
            }
            Ok(argv)
        }
    }

    fn expand(
        tmpl: &[String],
        params: &[crate::manifest::Param],
        vals: &HashMap<String, String>,
    ) -> Vec<String> {
        let mut out = Vec::new();
        for t in tmpl {
            let whole = t
                .strip_prefix('{')
                .and_then(|s| s.strip_suffix('}'))
                .filter(|n| !n.contains(['{', '}']));
            if let Some(name) = whole {
                if let Some(v) = vals.get(name) {
                    out.push(v.clone());
                }
                continue;
            }
            let mut s = t.clone();
            for p in params {
                s = s.replace(
                    &format!("{{{}}}", p.name),
                    vals.get(&p.name).map(String::as_str).unwrap_or(""),
                );
            }
            out.push(s);
        }
        out
    }
}

// ===== api =====
mod assets {
    include!(concat!(env!("OUT_DIR"), "/assets.rs"));
}

mod api {
    use crate::budget::{LISTING_TTL, MAX_QUERY};
    use crate::fail::{self, Error};
    use crate::harness::{self, Harness};
    use crate::session::{Channel, Frame, Inbox, Rig, Session};
    use crate::trace::{self, Trace, TraceId};
    use async_stream::stream;
    use axum::{
        Json,
        extract::{Path as UrlPath, Query, Request, State},
        http::{HeaderMap, HeaderValue, StatusCode},
        middleware::Next,
        response::{
            IntoResponse, Response,
            sse::{Event as SseEvent, KeepAlive, Sse},
        },
    };
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::ops::Deref;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{Instant, UNIX_EPOCH};
    use tokio::sync::Semaphore;
    use tokio::sync::broadcast::error::RecvError;
    use tokio_stream::Stream;

    pub struct Listing {
        at: Instant,
        body: Value,
    }

    pub struct Core {
        pub http: reqwest::Client,
        pub default: String,
        pub root: PathBuf,
        pub token: Option<String>,
        pub origin: Option<String>,
        pub model_gate: Arc<Semaphore>,
        pub tool_gate: Arc<Semaphore>,
        pub harnesses: Mutex<HashMap<String, Arc<Harness>>>,
        pub live: crate::session::Registry,
        pub listing: Mutex<Option<Listing>>,
    }

    #[derive(Clone)]
    pub struct App(Arc<Core>);

    impl App {
        pub fn new(core: Core) -> App {
            App(Arc::new(core))
        }

        async fn harness(&self, path: Option<&str>) -> Result<Arc<Harness>, String> {
            let key = match path {
                None => self.default.clone(),
                Some(p) => {
                    let abs = tokio::fs::canonicalize(p)
                        .await
                        .map_err(|e| format!("{p}: {e}"))?;
                    if !abs.starts_with(&self.root) {
                        return Err("harness must live inside the manifest directory".to_string());
                    }
                    abs.display().to_string()
                }
            };
            if let Some(h) = crate::lock(&self.harnesses).get(&key) {
                return Ok(h.clone());
            }
            let k = key.clone();
            let h = tokio::task::spawn_blocking(move || harness::compile(std::path::Path::new(&k)))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| format!("{e:#}"))?;
            let h = Arc::new(h);
            crate::lock(&self.harnesses).insert(key, h.clone());
            Ok(h)
        }
    }

    impl Deref for App {
        type Target = Core;

        fn deref(&self) -> &Core {
            &self.0
        }
    }

    #[derive(Deserialize)]
    pub struct RunReq {
        query: String,
        #[serde(default)]
        harness: Option<String>,
        #[serde(default)]
        hitch: Option<String>,
    }

    #[derive(Deserialize)]
    pub struct Since {
        #[serde(default)]
        since: u64,
    }

    #[derive(Deserialize)]
    pub struct Tail {
        #[serde(default)]
        tail: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct SayReq {
        trace_id: String,
        text: String,
        #[serde(default)]
        from: Option<String>,
    }

    fn to_sse(f: &Frame) -> SseEvent {
        SseEvent::default()
            .id(f.seq.to_string())
            .event(f.kind.clone())
            .data(f.body.clone())
    }

    fn frames(
        replay: Vec<Frame>,
        mut rx: tokio::sync::broadcast::Receiver<Frame>,
        since: u64,
    ) -> impl Stream<Item = Result<SseEvent, Infallible>> {
        stream! {
            let mut last = since;
            for f in replay {
                last = last.max(f.seq);
                yield Ok(to_sse(&f));
            }
            loop {
                let frame = match rx.recv().await {
                    Ok(f) if f.seq > last => { last = f.seq; f }
                    Ok(_) => continue,
                    Err(RecvError::Lagged(_)) => Frame::reload(),
                    Err(RecvError::Closed) => break,
                };
                yield Ok(to_sse(&frame));
            }
        }
    }

    fn sse(channel: Arc<Channel>, since: u64) -> Response {
        let rx = channel.subscribe();
        let replay = channel.snapshot(since);
        Sse::new(frames(replay, rx, since))
            .keep_alive(KeepAlive::default())
            .into_response()
    }

    async fn spawn(app: App, id: Option<String>, req: RunReq) -> fail::Result<Response> {
        if req.query.len() > MAX_QUERY {
            return Err(Error::TooLarge);
        }
        let harness = app
            .harness(req.harness.as_deref())
            .await
            .map_err(Error::BadRequest)?;
        let trace = match id {
            None => Trace::create().await?,
            Some(id) => {
                if crate::lock(&app.live).contains_key(&id) {
                    return Err(Error::Conflict);
                }
                Trace::open(id).await?
            }
        };
        let rig = Rig {
            http: app.http.clone(),
            model_gate: app.model_gate.clone(),
            tool_gate: app.tool_gate.clone(),
            live: app.live.clone(),
        };
        let session = Session::new(rig, harness, trace, req.query, req.hitch)?;
        let channel = session.channel.clone();
        tokio::spawn(session.run());
        Ok(sse(channel, 0))
    }

    pub async fn run(State(app): State<App>, Json(req): Json<RunReq>) -> fail::Result<Response> {
        spawn(app, None, req).await
    }

    pub async fn resume(
        State(app): State<App>,
        UrlPath(id): UrlPath<TraceId>,
        Json(req): Json<RunReq>,
    ) -> fail::Result<Response> {
        spawn(app, Some(id.0), req).await
    }

    pub async fn say(
        State(app): State<App>,
        Json(req): Json<SayReq>,
    ) -> fail::Result<&'static str> {
        if req.text.len() > MAX_QUERY {
            return Err(Error::TooLarge);
        }
        let inbox = crate::lock(&app.live)
            .get(&req.trace_id)
            .map(|h| h.inbox.clone());
        match inbox {
            Some(tx)
                if tx
                    .send(Inbox {
                        from: req.from,
                        text: req.text,
                    })
                    .is_ok() =>
            {
                Ok("ok")
            }
            _ => Err(Error::NotFound),
        }
    }

    pub async fn stop(
        State(app): State<App>,
        UrlPath(id): UrlPath<TraceId>,
    ) -> fail::Result<&'static str> {
        let live = crate::lock(&app.live);
        let handle = live.get(&id.0).ok_or(Error::NotFound)?;
        handle.stop.cancel();
        Ok("ok")
    }

    pub async fn events(
        State(app): State<App>,
        UrlPath(id): UrlPath<TraceId>,
        Query(q): Query<Since>,
    ) -> fail::Result<Response> {
        let channel = crate::lock(&app.live)
            .get(&id.0)
            .map(|h| h.channel.clone())
            .ok_or(Error::NotFound)?;
        Ok(sse(channel, q.since))
    }

    pub async fn sessions(State(app): State<App>) -> Json<Value> {
        let ids: Vec<String> = crate::lock(&app.live).keys().cloned().collect();
        Json(json!({"active": ids}))
    }

    pub async fn traces(State(app): State<App>) -> Json<Value> {
        if let Some(c) = crate::lock(&app.listing).as_ref()
            && c.at.elapsed() < LISTING_TTL
        {
            return Json(c.body.clone());
        }
        let mut rows: Vec<(u64, String)> = Vec::new();
        if let Ok(dir) = trace::traces_dir()
            && let Ok(mut rd) = tokio::fs::read_dir(&dir).await
        {
            while let Ok(Some(e)) = rd.next_entry().await {
                let name = e.file_name().to_string_lossy().to_string();
                if let Some(id) = name.strip_suffix(".yaml")
                    && trace::valid_id(id)
                {
                    let ts = e
                        .metadata()
                        .await
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    rows.push((ts, id.to_owned()));
                }
            }
        }
        rows.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
        rows.truncate(100);
        let body = json!(
            rows.iter()
                .map(|(ts, id)| json!({"id": id, "ts": ts}))
                .collect::<Vec<_>>()
        );
        *crate::lock(&app.listing) = Some(Listing {
            at: Instant::now(),
            body: body.clone(),
        });
        Json(body)
    }

    pub async fn trace_view(
        UrlPath(id): UrlPath<TraceId>,
        Query(q): Query<Tail>,
    ) -> fail::Result<Json<Value>> {
        let mut entries = trace::read_entries(&id.0).await?;
        if let Some(n) = q.tail
            && entries.len() > n
        {
            entries = entries.split_off(entries.len() - n);
        }
        Ok(Json(json!({"id": id.0, "entries": entries})))
    }

    pub async fn mathjax(UrlPath(path): UrlPath<String>) -> fail::Result<Response> {
        let bytes = crate::assets::asset(&path).ok_or(Error::NotFound)?;
        let content_type = if path.ends_with(".js") {
            "application/javascript"
        } else if path.ends_with(".woff") {
            "font/woff"
        } else {
            "application/octet-stream"
        };
        Ok((
            [
                ("content-type", content_type),
                ("cache-control", "public, max-age=31536000, immutable"),
            ],
            bytes,
        )
            .into_response())
    }

    pub async fn ui(State(app): State<App>) -> fail::Result<Response> {
        let html = tokio::fs::read_to_string(app.root.join("ui.html"))
            .await
            .map_err(|_| Error::NotFound)?;
        Ok(([("content-type", "text/html; charset=utf-8")], html).into_response())
    }

    pub async fn sedlo() -> impl IntoResponse {
        (
            [("content-type", "text/html; charset=utf-8")],
            include_str!("../assets/sedlo.html"),
        )
    }

    fn host_allowed(host: &str) -> bool {
        let name = match host.strip_prefix('[') {
            Some(rest) => rest.split(']').next().unwrap_or(""),
            None => host.split(':').next().unwrap_or(""),
        };
        matches!(name, "127.0.0.1" | "localhost" | "::1")
    }

    fn local_origin(origin: &str) -> bool {
        if origin == "null" {
            return true;
        }
        let Ok(url) = reqwest::Url::parse(origin) else {
            return false;
        };
        matches!(url.scheme(), "http" | "https")
            && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
            && url.origin().ascii_serialization() == origin
    }

    #[cfg(test)]
    mod cors_tests {
        use super::local_origin;

        #[test]
        fn local_origins_only() {
            for origin in [
                "null",
                "http://localhost:3000",
                "http://127.0.0.1:8080",
                "http://[::1]:3000",
            ] {
                assert!(local_origin(origin), "{origin}");
            }
            for origin in [
                "https://localhost.evil.example",
                "http://127.0.0.1.evil.example",
                "https://example.com",
                "http://localhost/path",
            ] {
                assert!(!local_origin(origin), "{origin}");
            }
        }
    }

    fn allow_origin(h: &mut HeaderMap, origin: &str) {
        if let Ok(v) = origin.parse() {
            h.insert("access-control-allow-origin", v);
        }
        h.insert("vary", HeaderValue::from_static("origin"));
        h.insert(
            "access-control-allow-methods",
            HeaderValue::from_static("GET, POST, OPTIONS"),
        );
        h.insert(
            "access-control-allow-headers",
            HeaderValue::from_static("content-type, authorization, last-event-id"),
        );
    }

    pub async fn guard(State(app): State<App>, req: Request, next: Next) -> Response {
        let host = req
            .headers()
            .get("host")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !host_allowed(host) {
            return Error::Forbidden.into_response();
        }

        let origin = req
            .headers()
            .get("origin")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let allowed = match (&app.origin, &origin) {
            (Some(want), Some(got)) if want == "*" || want == got => Some(if want == "*" {
                "*".to_string()
            } else {
                got.clone()
            }),
            (_, Some(got)) if local_origin(got) => Some(got.clone()),
            _ => None,
        };
        if req.method().as_str() == "OPTIONS" {
            let mut res = StatusCode::NO_CONTENT.into_response();
            if let Some(o) = &allowed {
                allow_origin(res.headers_mut(), o);
            }
            return res;
        }
        if req.uri().path().starts_with("/assets/mathjax-3.2.2/") {
            return next.run(req).await;
        }
        if let Some(tok) = &app.token {
            let got = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if got != format!("Bearer {tok}") {
                return Error::Unauthorized.into_response();
            }
        }
        let mut res = next.run(req).await;
        if let Some(o) = &allowed {
            allow_origin(res.headers_mut(), o);
        }
        res
    }
}

// ===== cli =====
mod cli {
    use std::env;
    use std::path::PathBuf;
    use std::process::exit;

    const HELP: &str = "\
uzda - run a model under a manifest

usage: uzda serve <manifest.yaml>

env:
  UZDA_ADDR          listen address (default 127.0.0.1:8080)
  UZDA_API_KEY       bearer token sent to the model endpoint
  UZDA_TOKEN         require this bearer token from participants
  UZDA_CORS_ORIGIN   additionally allow this web origin or * (local origins allowed by default)
";

    pub fn parse() -> Option<PathBuf> {
        let args: Vec<String> = env::args().skip(1).collect();
        match args.as_slice() {
            [flag] if flag == "-h" || flag == "--help" => {
                print!("{HELP}");
                None
            }
            [cmd, path] if cmd == "serve" => Some(PathBuf::from(path)),
            _ => {
                eprint!("{HELP}");
                exit(2);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use crate::api::{App, Core};
    use crate::session::Registry;
    use axum::{
        Router, middleware,
        routing::{get, post},
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tokio::sync::Semaphore;

    let Some(path) = cli::parse() else {
        return Ok(());
    };
    let path = std::fs::canonicalize(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let harness = match harness::compile(&path) {
        Ok(h) => Arc::new(h),
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };
    let root = harness.dir.clone();

    let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    })
    .with_no_client_auth();
    let http = reqwest::Client::builder()
        .tls_backend_preconfigured(tls)
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;

    let key = path.display().to_string();
    let live: Registry = Arc::new(Mutex::new(HashMap::new()));
    let app = App::new(Core {
        http,
        default: key.clone(),
        root,
        token: std::env::var("UZDA_TOKEN").ok().filter(|s| !s.is_empty()),
        origin: std::env::var("UZDA_CORS_ORIGIN")
            .ok()
            .filter(|s| !s.is_empty()),
        model_gate: Arc::new(Semaphore::new(1)),
        tool_gate: Arc::new(Semaphore::new(crate::budget::TOOL_SLOTS)),
        harnesses: Mutex::new(HashMap::from([(key, harness)])),
        live,
        listing: Mutex::new(None),
    });
    let router = Router::new()
        .route("/", get(api::ui))
        .route("/sedlo.html", get(api::sedlo))
        .route("/assets/mathjax-3.2.2/{*path}", get(api::mathjax))
        .route("/health", get(|| async { "ok" }))
        .route("/sessions", get(api::sessions))
        .route("/traces", get(api::traces))
        .route("/trace/{id}", get(api::trace_view))
        .route("/events/{id}", get(api::events))
        .route("/run", post(api::run))
        .route("/run/{id}", post(api::resume))
        .route("/say", post(api::say))
        .route("/stop/{id}", post(api::stop))
        .layer(middleware::from_fn_with_state(app.clone(), api::guard))
        .with_state(app);

    let addr = std::env::var("UZDA_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("uzda -> {addr}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::argv;
    use crate::trace::{Entry, UserText};

    fn round_trip(entries: &[Entry]) {
        let doc: String = entries.iter().map(|e| format!("{}\n", e.yaml())).collect();
        let back: Vec<Entry> = serde_saphyr::from_str(&doc).expect("emitted trace must parse");
        assert_eq!(back.len(), entries.len());
        for (a, b) in entries.iter().zip(&back) {
            assert_eq!(a.text(), b.text());
        }
    }

    #[test]
    fn trace_survives_hostile_text() {
        let nasty = [
            "", "123", "yes", "12:30", "http://x", "=", "-l", "a\n\n", "a\nb\n", "a\n\x00b",
            "\u{7f}", "\u{2028}", " lead", "trail ", "a #b", "a: b",
        ];
        let entries: Vec<Entry> = nasty.iter().map(|s| Entry::Result(s.to_string())).collect();
        round_trip(&entries);
    }

    #[test]
    fn hitch_entries_round_trip() {
        round_trip(&[Entry::User(UserText::From {
            from: "ops".to_string(),
            text: "check\nthe config".to_string(),
        })]);
    }

    #[test]
    fn argv_matches_shell_quoting() {
        let cases: &[(&str, &[&str])] = &[
            ("", &[]),
            ("-r -n", &["-r", "-n"]),
            ("\"my file.txt\"", &["my file.txt"]),
            ("'a b' c", &["a b", "c"]),
            ("a\"\"b", &["ab"]),
            ("\"\"", &[""]),
            ("x\\ y", &["x y"]),
            ("\"a\\\"b\"", &["a\"b"]),
        ];
        for (line, want) in cases {
            let got = argv::split_args(line).expect("well-formed line");
            assert_eq!(got, *want, "line: {line:?}");
        }
        for bad in ["'open", "\"open", "trailing\\"] {
            assert!(argv::split_args(bad).is_err(), "must reject {bad:?}");
        }
    }

    #[tokio::test]
    async fn resume_ignores_attempted_transfers() {
        use crate::harness::{Harness, History, Stage};
        use crate::trace::Trace;
        use std::collections::HashMap;
        let stage = || Stage {
            desc: String::new(),
            system: String::new(),
            schemas: Vec::new(),
            tools: HashMap::new(),
            skills: HashMap::new(),
            next: Vec::new(),
            history: History::default(),
        };
        let harness = Harness {
            dir: Default::default(),
            endpoint: String::new(),
            model: String::new(),
            key: String::new(),
            limits: Default::default(),
            start: "plan".into(),
            stages: HashMap::from([("plan".into(), stage()), ("execute".into(), stage())]),
        };
        let file = tokio::fs::File::open("Cargo.toml")
            .await
            .expect("open fixture");
        let mut trace = Trace {
            id: String::new(),
            file,
            entries: vec![Entry::Call("transfer_to_execute()".into())],

            broken: false,
        };
        assert_eq!(trace.stage(&harness), "plan");
        trace.entries.push(Entry::Stage("execute".into()));
        assert_eq!(trace.stage(&harness), "execute");
        round_trip(&trace.entries);
    }

    #[tokio::test]
    async fn skills_require_empty_arguments_and_valid_instruction() {
        use crate::exec::{self, Call, Effect};
        use crate::harness::{Harness, History, Stage};
        use std::{collections::HashMap, sync::Arc};
        let stage = Stage {
            desc: String::new(),
            system: String::new(),
            schemas: Vec::new(),
            tools: HashMap::new(),
            skills: HashMap::from([(
                "review".into(),
                "examples/prompts/document-review.md".into(),
            )]),
            next: Vec::new(),
            history: History::default(),
        };
        let harness = Arc::new(Harness {
            dir: Default::default(),
            endpoint: String::new(),
            model: String::new(),
            key: String::new(),
            limits: Default::default(),
            start: "default".into(),
            stages: HashMap::from([("default".into(), stage)]),
        });
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
        let call = Call {
            name: "review".into(),
            args: serde_json::Map::new(),
            err: None,
        };
        let result = exec::execute(gate.clone(), harness.clone(), "default".into(), call).await;
        assert!(matches!(result.effect, Effect::Skill(_, _)));
        let call = Call {
            name: "review".into(),
            args: serde_json::Map::from_iter([("extra".into(), serde_json::json!(1))]),
            err: None,
        };
        let result = exec::execute(gate, harness, "default".into(), call).await;
        assert!(!result.ok);
    }

    #[test]
    fn non_scalar_args_are_never_a_silent_noop() {
        let mut args = serde_json::Map::new();
        args.insert("args".to_string(), serde_json::json!(["-l"]));
        assert!(argv::args_line(&args).is_err());
        let empty = serde_json::Map::new();
        assert_eq!(argv::args_line(&empty).unwrap(), "");
    }
}
