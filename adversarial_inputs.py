
"""
Analyze a real image with adversarial attacks and create detailed visualizations.
Compares DeepFool, C&W L2, and PGD L2 attacks on a real image.
"""

import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchattacks
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

def load_image(image_path, transform):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor, filename):
    """Save a PyTorch tensor as an image."""
    # Convert tensor to numpy and denormalize
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # Clip to valid range
    image = np.clip(image, 0, 1)
    # Save image
    plt.imsave(filename, image)

def plot_comprehensive_comparison(results, save_path):
    """
    Create a comprehensive comparison visualization with all attacks.
    
    Args:
        results: List of dictionaries with attack results
        save_path: Path to save the visualization
    """
    n_attacks = len(results)
    fig = plt.figure(figsize=(20, 5 * n_attacks))
    gs = gridspec.GridSpec(n_attacks, 4, width_ratios=[1, 1, 1, 0.05])
    
    # Original image is the same for all attacks
    original_img = results[0]['original_image']
    original_np = original_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # Create a custom colormap for the perturbation heatmap
    cmap = LinearSegmentedColormap.from_list('perturbation', ['blue', 'white', 'red'])
    
    for i, result in enumerate(results):
        # Original image (only show in first row)
        if i == 0:
            ax0 = plt.subplot(gs[i, 0])
            ax0.imshow(original_np)
            ax0.set_title(f"Original: {result['original_label']}", fontsize=14)
            ax0.axis('off')
        else:
            ax0 = plt.subplot(gs[i, 0])
            ax0.imshow(original_np)
            ax0.set_title(f"Original", fontsize=14)
            ax0.axis('off')
        
        # Adversarial image
        ax1 = plt.subplot(gs[i, 1])
        adv_np = result['adversarial_image'].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        ax1.imshow(adv_np)
        ax1.set_title(f"{result['attack_name']}\nAdversarial: {result['adversarial_label']}", fontsize=14)
        ax1.axis('off')
        
        # Perturbation visualization (heatmap)
        ax2 = plt.subplot(gs[i, 2])
        pert = result['adversarial_image'] - original_img
        pert_np = pert.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        # Use L2 norm across channels to create a single channel heatmap
        pert_magnitude = np.sqrt(np.sum(pert_np**2, axis=2))
        # Scale for better visibility
        scale_factor = 5.0 / max(np.max(pert_magnitude), 1e-8)
        pert_magnitude = np.clip(pert_magnitude * scale_factor, 0, 1)
        
        im = ax2.imshow(pert_magnitude, cmap='hot')
        ax2.set_title(f"Perturbation Heatmap\nL2: {result['l2_distance']:.6f}, Time: {result['time']:.2f}s", fontsize=14)
        ax2.axis('off')
        
        # Add colorbar
        if i == 0:
            cax = plt.subplot(gs[i, 3])
            plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_perturbation_magnitudes(results, save_path):
    """Create a bar chart comparing perturbation magnitudes."""
    attack_names = [r['attack_name'] for r in results]
    l2_distances = [r['l2_distance'] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(attack_names, l2_distances, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylabel('L2 Distance (Lower is Better)', fontsize=14)
    plt.title('Perturbation Magnitude Comparison', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_computation_times(results, save_path):
    """Create a bar chart comparing computation times."""
    attack_names = [r['attack_name'] for r in results]
    times = [r['time'] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(attack_names, times, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
    
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.title('Attack Runtime Comparison', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_detailed_image_comparison(original, adversarial, perturbation, original_label, 
                                  adversarial_label, attack_name, l2_distance, save_path):
    """Create a detailed comparison for a single attack."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    orig_np = original.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original: {original_label}", fontsize=14)
    axes[0].axis('off')
    
    # Adversarial image
    adv_np = adversarial.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    axes[1].imshow(adv_np)
    axes[1].set_title(f"Adversarial: {adversarial_label}\nL2 Distance: {l2_distance:.6f}", fontsize=14)
    axes[1].axis('off')
    
    # Perturbation visualization with enhanced visibility
    pert_np = perturbation.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # Scale perturbation for visibility (multiply by 10 and center at 0.5)
    pert_display = pert_np * 10 + 0.5
    pert_display = np.clip(pert_display, 0, 1)
    axes[2].imshow(pert_display)
    axes[2].set_title(f"Perturbation (Enhanced 10x)", fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f"Attack: {attack_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    image_path = 'nita-anggraeni-goenawan-2JihaEVs8Dc-unsplash.jpg'
    results_dir = 'results_real_image'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load a pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Load class names
    try:
        with open('imagenet_classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Warning: ImageNet class names file not found. Using generic class labels.")
        class_names = [f"Class {i}" for i in range(1000)]
    
    # Load image
    try:
        image = load_image(image_path, transform).to(device)
    except FileNotFoundError:
        print(f"Error: Image {image_path} not found.")
        return
    
    # Get original prediction
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)
        original_label = class_names[predicted_idx.item()]
        confidence = probs[0, predicted_idx].item()
    
    print(f"Original prediction: {original_label} with {confidence:.4f} confidence")
    
    # Save the original image
    save_image(image, f"{results_dir}/original.png")
    
    # Define potential target classes that are visually distinct from a dog
    potential_targets = {
        "jellyfish": 107,
        "digital clock": 530,
        "volcano": 980,
        "scorpion": 69,
        "grand piano": 579,
        "teddy bear": 850,
        "butterfly": 326
    }
    
    # Choose a target class that's visually distinct from a dog
    target_class = "digital clock"
    target_idx = torch.tensor([potential_targets[target_class]]).to(device)
    print(f"Targeting class: {target_class} (index: {potential_targets[target_class]})")
    
    # Create attack instances with faster settings
    deepfool_attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    
    cw_attack = torchattacks.CW(model, c=5, steps=500, lr=0.01)
    cw_attack.set_mode_targeted_by_label()
    
    pgd_attack = torchattacks.PGDL2(model, eps=2.0, alpha=0.2, steps=50, random_start=True)
    pgd_attack.set_mode_targeted_by_label()
    
    # List of attacks to try
    attacks = [
        ("DeepFool", deepfool_attack),
        ("Carlini & Wagner L2", cw_attack),
        ("PGD L2", pgd_attack)
    ]
    
    # Store results for each attack
    results = []
    
    # Run each attack
    for attack_name, attack in attacks:
        print(f"\nRunning {attack_name} attack...")
        start_time = time.time()
        
        # Generate adversarial example (use target_idx for targeted attacks)
        if attack_name == "DeepFool":
            # DeepFool is not a targeted attack, so we use it normally
            adversarial_image = attack(image, predicted_idx)
        else:
            # For targeted attacks, we need to provide the target class index
            adversarial_image = attack(image, target_idx)
        
        # Calculate attack time
        attack_time = time.time() - start_time
        
        # Get prediction for adversarial image
        with torch.no_grad():
            adv_output = model(adversarial_image)
            adv_probs = torch.nn.functional.softmax(adv_output, dim=1)
            # print(f"Adversarial output probabilities: {adv_probs}")
            # Print top 5 predictions for adversarial image
            _, adv_predicted_idx = torch.topk(adv_probs, 5)
            print(f"Adversarial top 5 predictions: {[(class_names[idx.item()], adv_probs[0, idx].item()) for idx in adv_predicted_idx[0]]}")
            # Get the predicted label and confidence for the adversarial image
            _, adv_predicted_idx = torch.max(adv_output, 1)
            adversarial_label = class_names[adv_predicted_idx.item()]
            adv_confidence = adv_probs[0, adv_predicted_idx].item()
        
        # Calculate L2 distance
        l2_distance = torch.norm(adversarial_image - image, p=2).item()
        
        # Calculate perturbation
        perturbation = adversarial_image - image
        
        # Print results
        print(f"Attack completed in {attack_time:.2f} seconds")
        print(f"Adversarial prediction: {adversarial_label} with {adv_confidence:.4f} confidence")
        print(f"L2 distance: {l2_distance:.6f}")
        
        # Store result
        success = adv_predicted_idx != predicted_idx
        result = {
            "attack_name": attack_name,
            "time": attack_time,
            "original_label": original_label,
            "original_confidence": confidence,
            "adversarial_label": adversarial_label,
            "adversarial_confidence": adv_confidence,
            "l2_distance": l2_distance,
            "success": success,
            "original_image": image,
            "adversarial_image": adversarial_image
        }
        results.append(result)
        
        # Create and save detailed comparison for this attack
        plot_detailed_image_comparison(
            image, adversarial_image, perturbation, 
            original_label, adversarial_label, 
            attack_name, l2_distance,
            f"{results_dir}/detailed_{attack_name.replace(' ', '_').replace('&', 'and')}.png"
        )
        
        # Save the adversarial image
        save_image(adversarial_image, f"{results_dir}/adversarial_{attack_name.replace(' ', '_').replace('&', 'and')}.png")
    
    # Create comprehensive visualization with all attacks
    plot_comprehensive_comparison(results, f"{results_dir}/comprehensive_comparison.png")
    
    # Create bar charts for metrics
    plot_perturbation_magnitudes(results, f"{results_dir}/perturbation_comparison.png")
    plot_computation_times(results, f"{results_dir}/computation_time_comparison.png")
    
    # Print comparison table
    print("\n\n---- COMPARISON OF ATTACKS ----")
    print(f"{'Attack':<20} {'Success':<8} {'L2 Distance':<12} {'Time (s)':<10} {'Original':<15} {'Adversarial':<15}")
    print("-" * 85)
    
    for result in results:
        success_str = "Yes" if result["success"] else "No"
        print(f"{result['attack_name']:<20} {success_str:<8} {result['l2_distance']:<12.6f} {result['time']:<10.2f} {result['original_label']:<15} {result['adversarial_label']:<15}")
    
    # Identify the best attack (lowest L2 distance among successful attacks)
    successful_attacks = [result for result in results if result["success"]]
    
    if successful_attacks:
        best_attack = min(successful_attacks, key=lambda x: x["l2_distance"])
        print(f"\nMost imperceptible successful attack: {best_attack['attack_name']} with L2 distance of {best_attack['l2_distance']:.6f}")
    else:
        print("\nNo successful attacks found.")
    
    print(f"\nAll results saved in the '{results_dir}' directory.")

if __name__ == "__main__":
    main()
