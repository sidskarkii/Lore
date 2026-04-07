"""Take a screenshot and save to debug/sc.png for Claude to read."""
import sys
from pathlib import Path
from PIL import ImageGrab

out = Path(__file__).parent / "sc.png"
img = ImageGrab.grab(bbox=(2560, 0, 4480, 1080), all_screens=True)
img.save(out)
print(f"saved {img.size[0]}x{img.size[1]} to {out}")
