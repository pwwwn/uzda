use std::{env, fs, path::Path};

fn collect(dir: &Path, base: &Path, out: &mut String) {
    let entries = fs::read_dir(dir).expect("read bundled MathJax directory");
    for entry in entries {
        let path = entry.expect("read asset entry").path();
        if path.is_dir() {
            collect(&path, base, out);
        } else {
            let name = path
                .strip_prefix(base)
                .expect("asset path")
                .to_string_lossy();
            let absolute = fs::canonicalize(&path).expect("canonical asset path");
            out.push_str(&format!(
                "{:?} => Some(include_bytes!({:?}) as &'static [u8]),\n",
                name,
                absolute.to_string_lossy()
            ));
        }
    }
}

fn main() {
    let base = Path::new("assets/mathjax-3.2.2");
    println!("cargo:rerun-if-changed={}", base.display());
    let mut source =
        String::from("pub fn asset(path: &str) -> Option<&'static [u8]> { match path {\n");
    collect(base, base, &mut source);
    source.push_str("_ => None, } }\n");
    fs::write(
        Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR")).join("assets.rs"),
        source,
    )
    .expect("write asset index");
}
