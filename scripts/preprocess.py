import json
from datetime import datetime, timedelta
from pathlib import Path

class Preprocessor:
    def __init__(self, input_dir, output_dir, target_user=None, context_window=5, time_gap=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_user = target_user
        self.context_window = context_window
        self.time_gap = time_gap

    def load_json_files(self):
        """Load all JSON files from the input directory."""
        json_files = list(self.input_dir.glob("*.json"))
        data = []
        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data.append(json.load(f))
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading file {file}: {e}")
        return data

    def is_unprompted_message(self, current_msg, last_msg):
        """Determine if the current message is unprompted based on time gap."""
        current_time = datetime.strptime(current_msg['timestamp'], "%m/%d/%Y %I:%M %p")
        last_time = datetime.strptime(last_msg['timestamp'], "%m/%d/%Y %I:%M %p")
        return (current_time - last_time) > timedelta(minutes=self.time_gap)

    def get_context(self, messages, current_index):
        """Build the context for the current message based on the context window and time gap threshold."""
        context = []
        current_msg = messages[current_index]

        # Treat the first message in chat log as unprompted
        if current_index == 0:
            return "Start a new topic."
        
        # Loop backwards from the message just before the current message
        for i in range(current_index - 1, max(0, current_index - self.context_window) - 1, -1):
            last_msg = messages[i]
            if self.is_unprompted_message(current_msg, last_msg):
                continue  # Skip older messages that exceed the time gap
            context.insert(0, f"{last_msg['speaker']}: {last_msg['message']}")  # Add to the front of the context list

        return "\n".join(context) if context else "Start a new topic."

    def filter_user_messages(self, chat_logs):
        """Extract messages and their contexts, optionally filtering by target user."""
        fine_tune_data = []
        for chat in chat_logs:
            messages = chat.get("messages", [])
            for i, msg in enumerate(messages):
                if not self.validate_message(msg):
                    print(f"Skipping invalid message at index {i}: {msg}")
                    continue

                if self.target_user and msg["speaker"] != self.target_user:
                    continue  # Skip messages not from the target user

                context = self.get_context(messages, i)
                fine_tune_data.append({
                    "instruction": context,
                    "response": msg["message"]
                })
        return fine_tune_data

    def validate_message(self, msg):
        """Ensure the message contains required fields."""
        required_keys = {"timestamp", "speaker", "message"}
        return all(key in msg for key in required_keys)

    def save_data(self, data):
        """Save transformed data to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "fine_tune_dataset.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def run(self):
        """Execute the preprocessing pipeline."""
        print("Loading JSON files...")
        raw_data = self.load_json_files()
        print(f"Loaded {len(raw_data)} files.")

        print("Transforming data...")
        transformed_data = self.filter_user_messages(raw_data)

        print("Saving fine-tuning dataset...")
        self.save_data(transformed_data)
        print(f"Dataset saved to {self.output_dir / 'fine_tune_dataset.json'}")

def main():
    input_dir = "../data/processed"
    output_dir = "../data/fine_tune_ready"
    target_user = None  # Set to None to include all users

    preprocessor = Preprocessor(input_dir, output_dir, target_user, context_window=5, time_gap=5)
    preprocessor.run()

if __name__ == "__main__":
    main()