# DiscordMeLLM
Turn your Discord chat logs into a fine-tuned LLM Discord Chat Bot!

# Overview
This repository provides tools to:
- Preprocess parsed chat data for fine-tuning.
- Fine-tune a LLM using QLoRA.
- Deploy a Discord bot powered by the fine-tuned model.

The  fine tuning and inference scripts are designed to support various models, configurations, and parameter choices to accommodate different hardware capabilities and use cases. The default settings, including the use of `meta-llama/Llama-2-7b-chat-hf` as the base model and specific memory-saving optimizations, were selected to efficiently fine-tune and deploy the model on a 3070 Ti GPU with 8GB VRAM. These optimizations include 4-bit quantization (QLoRA), gradient checkpointing, a batch size of 1, and offloading model layers to CPU when necessary.



# Usage

## 1. Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to export your Discord chat logs.

## 2. Use the [DiscordLogParser](https://github.com/micogrande/DiscordLogParser) to parse your Discord chat logs:
Download and set up the parser from the linked repository and run the parser on your exported chat logs to create a `parsed_log.json` file.
        
Example command:
```bash
python scripts/discord_log_parser.py <input_log_file> data/processed/parsed_log.json
```        


## 3. Preprocess Logs
Preprocess the parsed logs into instruction response pairs based on the selected user of your choice to create a dataset for fine-tuning:
```bash
python scripts/preprocess.py
```
- **Input**: `data/processed/parsed_log.json`
- **Output**: `data/fine_tune_ready/fine_tune_dataset.json`

Target User: Update the target_user parameter to specify the username of interest. By default, it is set to `None`.

## 4. Download the base model
   ```bash
   python scripts/download_model.py
   ```

## 5. Fine-Tune the Model
Fine-tune the language model using the preprocessed dataset:
```bash
python scripts/fine_tune.py
```
- **Configuration**: `config/model_config.json`
- **Output**: Fine-tuned model saved in `models/fine_tuned/`


## 6. Set up the environment variables:
   - Create a `.env` file in the project root and add your Discord bot token:
     ```plaintext
     DISCORD_TOKEN=your_discord_bot_token
     ```

## 7. Add the bot to your discord server
Use [Discord Developer Portal](https://discord.com/developers/applications)

## 8. Run the Bot
Deploy the fine-tuned model as a Discord bot:
```bash
python scripts/discord_bot.py
```
- The bot listens for the `!ask` command in your Discord server and generates responses using the fine-tuned model.