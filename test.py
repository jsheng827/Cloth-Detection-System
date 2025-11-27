import torch

# Load your .pth.tar-60 checkpoint
#checkpoint = torch.load('model/osnet1.0-market-duke.tar-60', weights_only=False)  # Use your file path here

# Save only the state_dict (model weights) to a .pth file
#torch.save(checkpoint['state_dict'], 'osnet1.0-market-duke-trained.pth')
import deep_sort_realtime
import os

    # Get the directory where the deep_sort_realtime package is installed
package_path = os.path.dirname(deep_sort_realtime.__file__)
print(f"DeepSORT package path: {package_path}")