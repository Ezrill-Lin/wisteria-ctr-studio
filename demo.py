"""
Wisteria CTR Studio - Demo Script

This script demonstrates CTR prediction using SiliconSampling personas.

Usage:
    # Text ad prediction
    python demo.py --ad "Special 0% APR credit card offer for travel rewards" --population-size 100
    
    # Image ad prediction (URL)
    python demo.py --image-url "https://example.com/ad-image.jpg" --population-size 100
    
    # Image ad prediction (local file)
    python demo.py --image-path "path/to/ad-image.jpg" --population-size 100
    
    # Specify persona version and strategy
    python demo.py --ad "Shop our new summer collection!" --persona-version v2 --persona-strategy wpp --population-size 200
    
    # Save detailed results to JSON
    python demo.py --ad "Premium noise-canceling headphones on sale" --output results.json --population-size 100
"""

import argparse
import time

from CTRPrediction import CTRPredictor
from CTRPrediction.utils import encode_image_to_data_url
from display_utils import print_result, print_separator, save_result_json


def main():
    parser = argparse.ArgumentParser(
        description="CTR prediction using SiliconSampling personas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text ad prediction
  python demo.py --ad "Special offer: 50%% off premium membership!"
  
  # Image ad prediction (URL)
  python demo.py --image-url "https://example.com/ad-image.jpg"
  
  # Image ad prediction (local file)
  python demo.py --image-path "ads/banner.jpg"
  
  # Use v2 personas with WPP strategy
  python demo.py --ad "New eco-friendly product line" --persona-version v2 --persona-strategy wpp
  
  # Save results
  python demo.py --ad "Sign up today!" --output my_results.json
        """
    )
    
    # Ad content arguments (mutually exclusive)
    ad_group = parser.add_mutually_exclusive_group(required=True)
    ad_group.add_argument(
        "--ad",
        help="Advertisement text content"
    )
    ad_group.add_argument(
        "--image-url",
        help="Advertisement image URL"
    )
    ad_group.add_argument(
        "--image-path",
        help="Path to local advertisement image file"
    )
    
    # Persona configuration
    parser.add_argument(
        "--persona-version",
        choices=["v1", "v2"],
        default="v2",
        help="Persona version to use (default: v2)"
    )
    
    parser.add_argument(
        "--persona-strategy",
        choices=["random", "wpp", "ipip"],
        default="random",
        help="Persona generation strategy (default: random)"
    )
    
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Number of personas to evaluate (default: 100)"
    )
    
    # Platform configuration
    parser.add_argument(
        "--ad-platform",
        choices=["facebook", "tiktok", "amazon", "instagram", "youtube"],
        default="facebook",
        help="Platform where ad is shown (default: facebook)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file (optional)"
    )
    
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=20,
        help="Number of concurrent API requests (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Process image input
    image_url = None
    if args.image_url:
        image_url = args.image_url
    elif args.image_path:
        try:
            print(f"📁 Encoding local image: {args.image_path}")
            image_url = encode_image_to_data_url(args.image_path)
            print(f"✓ Image encoded successfully ({len(image_url)} bytes)")
        except Exception as e:
            print(f"\n❌ Error encoding image: {e}")
            return
    
    # Print configuration
    print("\n" + "="*80)
    print("WISTERIA CTR STUDIO - DEMO")
    print("="*80)
    print(f"\n📢 Advertisement:")
    if args.ad:
        print(f"   Type: Text Ad")
        print(f"   Content: \"{args.ad}\"")
    else:
        print(f"   Type: Image Ad")
        if args.image_url:
            print(f"   Source: Remote URL")
            print(f"   Image URL: {args.image_url}")
        else:
            print(f"   Source: Local File")
            print(f"   Image Path: {args.image_path}")
    print(f"\n⚙️  Configuration:")
    print(f"   Platform: {args.ad_platform}")
    print(f"   Population Size: {args.population_size:,}")
    print(f"   Persona Version: {args.persona_version}")
    print(f"   Persona Strategy: {args.persona_strategy}")
    print("="*80 + "\n")
    
    # Create predictor
    try:
        predictor = CTRPredictor(
            persona_version=args.persona_version,
            persona_strategy=args.persona_strategy,
            concurrent_requests=args.concurrent_requests
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease generate personas first by running:")
        print(f"   cd SiliconSampling/personas{'_v2' if args.persona_version == 'v2' else ''}")
        print(f"   python generate_personas{'_v2' if args.persona_version == 'v2' else ''}.py --strategy {args.persona_strategy} --sample-size {args.population_size}")
        return
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Run prediction
    start_time = time.time()
    
    try:
        if args.ad:
            result = predictor.predict(
                ad_text=args.ad,
                population_size=args.population_size,
                ad_platform=args.ad_platform
            )
        else:
            result = predictor.predict(
                image_url=image_url,
                population_size=args.population_size,
                ad_platform=args.ad_platform
            )
    except Exception as e:
        print(f"\n❌ Prediction failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n⏱️  Total Time: {elapsed_time:.2f} seconds")
    print(f"   Average Time per Persona: {elapsed_time / args.population_size:.2f} seconds\n")
    
    print_result(result)
    
    # Save to file if requested
    if args.output:
        save_result_json(result, args.output)
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()
