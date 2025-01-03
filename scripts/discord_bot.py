from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import discord
from discord.ext import commands
import torch
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the .env file from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class SimpleBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)

        # Configure device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer with 4-bit quantization
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16  # Match input type for performance
            )

            logger.info("Loading base model and tokenizer...")
            base_model_path = "../models/base_model"  # Replace with your base model path
            adapter_path = "../models/fine_tuned/final"  # Replace with your adapter path

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                offload_folder="offload"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

            # Set padding token to eos_token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading the LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("Model and LoRA adapter loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            raise

    async def setup_hook(self):
        await self.add_cog(SimpleCog(self))


class SimpleCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def ask(self, ctx, *, question: str):
        """Ask a question to the model"""
        if not question.strip():
            await ctx.send("Please provide a valid question.")
            return

        async with ctx.typing():
            try:
                # Tokenize the user's input with attention mask
                inputs = self.bot.tokenizer(
                    question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.bot.device)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Generate the response
                with torch.no_grad():
                    outputs = self.bot.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,  # Explicitly pass the attention mask
                        max_length=300,  # Adjust as needed
                        temperature=0.7,
                        top_p=0.9,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.bot.tokenizer.eos_token_id
                    )

                # Decode the model's response
                response = self.bot.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Send response in chunks if it's too long
                if len(response) > 2000:
                    chunks = [response[i:i+1990] for i in range(0, len(response), 1990)]
                    for chunk in chunks:
                        await ctx.send(chunk)
                else:
                    await ctx.send(response)

            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                await ctx.send("Sorry, something went wrong while processing your request.")

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(f'Bot is ready and logged in as {self.bot.user}!')


def main():
    # Create and run the bot
    bot_token = os.getenv('DISCORD_TOKEN')  # Load token from .env file
    if not bot_token:
        raise ValueError("Discord token not found in environment variables!")

    bot = SimpleBot()
    try:
        bot.run(bot_token)
    except Exception as e:
        logger.error(f"Failed to run the bot: {e}")


if __name__ == "__main__":
    main()