mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use half::f16;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    #[cfg(feature = "story")]
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    #[cfg(feature = "chat")]
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    #[cfg(feature = "fp16")]
    let llama = model::Llama::<f16>::from_safetensors(&model_dir);
    #[cfg(not(feature = "fp16"))]
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    #[cfg(feature = "story")] {
        let input = "Once upon a time";
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        print!("\n{}", input);
        let output_ids = llama.generate(
            input_ids,
            500,
            0.8,
            30,
            1.,
        );
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
    #[cfg(feature = "chat")]
    llama.chat_generate(
        tokenizer,
        500,
        0.8,
        30,
        1.
    );
}
