import os
import dotenv
dotenv.load_dotenv()

num_classes = os.getenv("NUM_CLASSES")
height = os.getenv("TARGET_HEIGHT")
width = os.getenv("TARGET_WIDTH")
batch_size = os.getenv("BATCH_SIZE")

print(num_classes, height, width, batch_size)