import os
import time
import random
import requests
import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from io import BytesIO
# Import your modules
from media_pipe_estimator import has_pose
from MobileNetv2_model import is_human_image

def setup_stealth_driver():
    """Setup a Selenium driver with anti-detection measures"""
    options = Options()
    
    # Comment this out to run in visible mode for debugging
    options.add_argument("--headless=new")
    
    # Window size (important to mimic a real browser)
    options.add_argument("--window-size=1920,1080")
    
    # Use common user agent for MacOS
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Anti-bot detection measures
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Reduce resource usage
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    
    # Language settings that won't trigger unusual Google variants
    options.add_argument("--lang=en-US")
    
    # Timezone to appear like a regular US user
    options.add_argument("--timezone=America/Los_Angeles")
    
    # Create and configure the driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Additional stealth settings via JavaScript
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {
        "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    
    # Set some fake permissions to appear more like a real browser
    driver.execute_cdp_cmd("Browser.grantPermissions", {
        "origin": "https://www.google.com",
        "permissions": ["geolocation", "notifications"]
    })
    
    return driver

def natural_scrolling(driver, scroll_count=3):
    """Scrolls in a more human-like manner with random pauses"""
    for i in range(scroll_count):
        # Random scroll amount
        scroll_amount = random.randint(300, 800)
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        
        # Random pause to simulate human behavior
        time.sleep(random.uniform(1.0, 3.5))
        
        # Sometimes move the mouse (via JavaScript)
        if random.random() > 0.7:
            x, y = random.randint(100, 700), random.randint(100, 500)
            driver.execute_script(f"document.body.dispatchEvent(new MouseEvent('mousemove', {{clientX: {x}, clientY: {y}, bubbles: true}}))")
            time.sleep(random.uniform(0.1, 0.5))

def scrape_google_images(query, max_images=10):
    """Scrape Google Images using a stealthy Selenium setup"""
    driver = setup_stealth_driver()
    
    # Add some variety to the search query to avoid detection patterns
    query_modifiers = ["", "pictures", "photos", "images"]
    selected_modifier = random.choice(query_modifiers)
    if selected_modifier:
        full_query = f"{query} {selected_modifier}"
    else:
        full_query = query
    
    # Base URL for Google Images search
    search_url = f"https://www.google.com/search?q={full_query.replace(' ', '+')}&tbm=isch"
    print(f"Opening URL: {search_url}")
    
    # First visit Google homepage to set cookies
    driver.get("https://www.google.com")
    time.sleep(random.uniform(1.0, 3.0))
    
    # Now go to the search URL
    driver.get(search_url)
    
    # Random pause like a human would after page load
    time.sleep(random.uniform(2.0, 4.0))
    
    # Accept cookies if the dialog appears
    try:
        cookie_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Accept all') or contains(., 'I agree') or contains(., 'Accept')]")
        if cookie_buttons:
            cookie_buttons[0].click()
            time.sleep(random.uniform(1.0, 2.0))
    except Exception as e:
        print(f"No cookie prompt or error: {e}")
    
    # Scroll naturally to load more images
    natural_scrolling(driver, scroll_count=5)
    
    # Get image elements
   # Get image elements
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
    print(f"Found {len(image_elements)} image elements")
    
    # Filter out very small images which are likely icons
    valid_images = []
    for img in image_elements:
        try:
            width = int(img.get_attribute("width") or 0)
            height = int(img.get_attribute("height") or 0)
            if width > 60 and height > 60:
                valid_images.append(img)
        except:
            continue
    
    print(f"Found {len(valid_images)} valid-sized images")
    
    # Extract image URLs
    image_urls = []
    image_data_uris = []
    for img in valid_images:
        try:
            src = img.get_attribute("src")
            if src and "http" in src and not src.startswith("data:"):
                # Avoid very small images and icons
                if "gstatic" not in src and "google" not in src:
                    image_urls.append(src)
                    print(f"Found image URL: {src[:50]}...")
                    
                    # Sometimes pause between extractions
                    if random.random() > 0.8:
                        time.sleep(random.uniform(0.2, 1.0))
            elif src and src.startswith('data:image'):
                # This is a data URI image
                image_data_uris.append(src)
                print(f"Found data URI image (length: {len(src)})")
                
        except Exception as e:
            print(f"Error extracting image URL: {e}")
    
    # Limit to the requested number of images
    image_urls = image_urls[:max_images]
    image_data_uris = image_data_uris[:max_images]
    print(f"Total unique images: {len(image_urls) + len(image_data_uris)}")
    
    # Close the browser after random delay
    time.sleep(random.uniform(1.0, 2.0))
    driver.quit()
    
    return image_urls, image_data_uris

def download_image(url, folder, filename):
    """Download an image from a URL"""
    try:
        # Random delay between downloads to avoid rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        # Use a rotating set of user agents
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15"
        ]
        
        headers = {
            "User-Agent": random.choice(user_agents),
            "Referer": "https://www.google.com/",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=10, headers=headers)
        
        # Check if request is successful
        if response.status_code != 200:
            print(f"⚠️ Skipping {url} (Status code: {response.status_code})")
            return None

        # Read image data
        img_data = BytesIO(response.content)

        # Validate image format
        try:
            img = Image.open(img_data)
            # Check image dimensions
            width, height = img.size
            if width < 100 or height < 100:
                print(f"⚠️ Skipping {url} (Too small: {width}x{height})")
                return None
                
            # Save image
            image_path = os.path.join(folder, filename)
            img = img.convert("RGB")  # Convert for consistency
            img.save(image_path, format="JPEG")  # Save as JPEG
            print(f"Successfully downloaded and saved to {image_path}")
            return image_path
        except Exception as e:
            print(f"Invalid image format {url}: {e}")
            return None

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None
    
def save_data_uri(data_uri, image_path):
    if ',' in data_uri:
        # Split at the comma to get just the base64 data
        header, encoded = data_uri.split(',', 1)
        # Decode the base64 data
        binary_data = base64.b64decode(encoded)
        # Write to file
        with open(image_path, 'wb') as f:
            f.write(binary_data)
        print(f"Image saved to {image_path}")
    else:
        print("Invalid data URI format")

def scrape_and_filter_images(query="human side stance", save_folder="posture_images", max_images=1000):
    """Scrape, download, and filter images from Google Images"""
    os.makedirs(save_folder, exist_ok=True)
    print(f"Scraping images for query: '{query}'...")
    
    # Try with different queries until we get results
    image_urls, image_data_uris = scrape_google_images(query, max_images)
    
    valid_images = 0
    if image_urls:
        for i, url in enumerate(image_urls):
            print(f"Processing image {i+1}/{len(image_urls)}")
            image_path = download_image(url, save_folder, f"image_{i}.jpg")
            
            # Check if the image download was successful before proceeding
            if image_path:
                try:
                    # Use your custom ML models to filter
                    if is_human_image(image_path) and has_pose(image_path):
                        print(f"Saved: {image_path}")
                        valid_images += 1
                    else:
                        if os.path.exists(image_path):
                            os.remove(image_path)  # Delete if not human
                        print(f"Deleted: {image_path} (Not a human or no pose detected)")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    if os.path.exists(image_path):
                        os.remove(image_path)
            else:
                print(f"Failed to process: {url}")
    
    # Process data URI images
    for i, data_uri in enumerate(image_data_uris):
        print(f"Processing data URI image {i+1}/{len(image_data_uris)}")
        image_path = os.path.join(save_folder, f"data_image_{i}.jpg")
        save_data_uri(data_uri, image_path)
        
        # Check if the image was saved successfully before proceeding
        if os.path.exists(image_path):
            try:
                # Use your custom ML models to filter
                if is_human_image(image_path)== True:
                    print(f"Is human: {image_path}")
                    # if has_pose(image_path) == True:
                    #     print(f"Has pose, image saved as: {image_path}")

                    valid_images += 1
                else:
                    if os.path.exists(image_path):
                        os.remove(image_path)  # Delete if not human
                    print(f"Deleted: {image_path} (Not a human or no pose detected)")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                if os.path.exists(image_path):
                    os.remove(image_path)
    
    print(f"Total Valid Images: {valid_images}")

if __name__ == "__main__":
    scrape_and_filter_images()