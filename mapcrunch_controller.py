import time
from typing import Dict, Optional, List

import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from config import MAPCRUNCH_URL, SELECTORS, DATA_COLLECTION_CONFIG


class MapCrunchController:
    def __init__(self, headless: bool = False):
        # Try to initialize ChromeDriver with version 137 (your current Chrome version)
        try:
            # Create fresh ChromeOptions for first attempt
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-web-security")
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            if headless:
                options.add_argument("--headless=new")

            self.driver = uc.Chrome(options=options, use_subprocess=True, version_main=137)
            print("✅ ChromeDriver initialized successfully with version 137")
        except Exception as e:
            print(f"Failed with version 137: {e}")
            try:
                # Create fresh ChromeOptions for fallback attempt
                options = uc.ChromeOptions()
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--disable-web-security")
                options.add_argument("--disable-features=VizDisplayCompositor")
                options.add_argument("--disable-blink-features=AutomationControlled")
                
                if headless:
                    options.add_argument("--headless=new")

                # Fallback to auto-detection
                self.driver = uc.Chrome(options=options, use_subprocess=True)
                print("✅ ChromeDriver initialized successfully with auto-detection")
            except Exception as e2:
                print(f"Failed with auto-detection: {e2}")
                raise Exception(f"Could not initialize ChromeDriver. Please update Chrome or check compatibility. Errors: {e}, {e2}")
        
        self.wait = WebDriverWait(self.driver, 10)
        
        # Inject browser detection bypass script
        try:
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                    Object.defineProperty(window, 'badBrowser', {
                      value: 0,
                      writable: false,
                      configurable: false
                    });
                    window.alert = function() {};
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    });
                """
                },
            )
        except Exception as e:
            print(f"Warning: Could not inject browser detection script: {e}")

        # Load MapCrunch
        for retry in range(3):
            try:
                self.driver.get(MAPCRUNCH_URL)
                time.sleep(3)
                print("✅ MapCrunch loaded successfully")
                break
            except Exception as e:
                if retry == 2:
                    raise e
                print(f"Failed to load MapCrunch, retry {retry + 1}/3: {e}")
                time.sleep(2)

    def setup_clean_environment(self):
        """
        Minimal environment setup using hideLoc() and hiding major UI.
        """
        self.driver.execute_script("if(typeof hideLoc === 'function') hideLoc();")
        self.driver.execute_script("""
            const menu = document.querySelector('#menu');
            if (menu) menu.style.display = 'none';
                                   
            const social = document.querySelector('#social');
            if (social) social.style.display = 'none';
                                   
            const googleImg = document.querySelector('img[alt="Google"]');
            if (googleImg && googleImg.parentElement) {
                googleImg.parentElement.style.display = 'none';
            }

            const topBar = document.querySelector('#topbar');
            if (topBar) topBar.style.display = 'none';

            const bottomBox = document.querySelector('#bottom-box');
            if (bottomBox) bottomBox.style.display = 'none';
            
            const infoFirstView = document.querySelector('#info-firstview');
            if (infoFirstView) infoFirstView.style.display = 'none';

            const controlsToHide = document.querySelectorAll('.gm-style-cc'); controlsToHide.forEach(el => { el.style.display = 'none'; });
            const keyboardButton = document.querySelector('button[aria-label="Keyboard shortcuts"]'); if (keyboardButton) { keyboardButton.style.display = 'none'; }


        """)

    def label_arrows_on_screen(self):
        """Overlays 'UP' and 'DOWN' labels on the navigation arrows."""
        try:
            pov = self.driver.execute_script("return window.panorama.getPov();")
            links = self.driver.execute_script("return window.panorama.getLinks();")
        except Exception:
            return

        if not links or not pov:
            return

        current_heading = pov["heading"]
        forward_link = None
        backward_link = None

        # This logic is identical to your existing `move` function
        # to ensure stylistic and behavioral consistency.
        min_forward_diff = 360
        for link in links:
            diff = 180 - abs(abs(link["heading"] - current_heading) - 180)
            if diff < min_forward_diff:
                min_forward_diff = diff
                forward_link = link

        target_backward_heading = (current_heading + 180) % 360
        min_backward_diff = 360
        for link in links:
            diff = 180 - abs(abs(link["heading"] - target_backward_heading) - 180)
            if diff < min_backward_diff:
                min_backward_diff = diff
                backward_link = link

        js_script = """
            document.querySelectorAll('.geobot-arrow-label').forEach(el => el.remove());
            document.querySelectorAll('path[data-geobot-modified]').forEach(arrow => {
                arrow.setAttribute('transform', arrow.getAttribute('data-original-transform') || '');
                arrow.removeAttribute('data-geobot-modified');
                arrow.removeAttribute('data-original-transform');
            });

            const modifyAndLabelArrow = (panoId, labelText, color) => {
                const arrowElement = document.querySelector(`path[pano="${panoId}"]`);
                if (!arrowElement) return;

                const originalTransform = arrowElement.getAttribute('transform') || '';
                arrowElement.setAttribute('data-original-transform', originalTransform);
                arrowElement.setAttribute('transform', `${originalTransform} scale(1.8)`);
                arrowElement.setAttribute('data-geobot-modified', 'true');

                const rect = arrowElement.getBoundingClientRect();
                const label = document.createElement('div');
                label.className = 'geobot-arrow-label';
                label.style.position = 'fixed';
                label.style.left = `${rect.left + rect.width / 2}px`;
                label.style.top = `${rect.top - 45}px`;
                label.style.transform = 'translateX(-50%)';
                label.style.padding = '5px 15px';
                label.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                label.style.color = color;
                label.style.borderRadius = '8px';
                label.style.fontSize = '28px';
                label.style.fontWeight = 'bold';
                label.style.zIndex = '99999';
                label.style.pointerEvents = 'none';
                label.innerText = labelText;
                document.body.appendChild(label);
            };

            const forwardPano = arguments[0];
            const backwardPano = arguments[1];

            if (forwardPano) {
                modifyAndLabelArrow(forwardPano, 'UP', '#76FF03');
            }
            if (backwardPano && backwardPano !== forwardPano) {
                modifyAndLabelArrow(backwardPano, 'DOWN', '#F44336');
            }
        """

        forward_pano = forward_link["pano"] if forward_link else None
        backward_pano = backward_link["pano"] if backward_link else None

        self.driver.execute_script(js_script, forward_pano, backward_pano)
        time.sleep(0.2)

    def get_available_actions(self) -> List[str]:
        """
        Checks for movement links via JavaScript.
        """
        base_actions = ["PAN_LEFT", "PAN_RIGHT", "GUESS"]
        links = self.driver.execute_script("return window.panorama.getLinks();")
        if links and len(links) > 0:
            base_actions.extend(["MOVE_FORWARD", "MOVE_BACKWARD"])
        return base_actions

    def get_test_available_actions(self) -> List[str]:
        """
        Checks for movement links via JavaScript.
        """
        base_actions = ["PAN_LEFT", "PAN_RIGHT"]
        links = self.driver.execute_script("return window.panorama.getLinks();")
        if links and len(links) > 0:
            base_actions.extend(["MOVE_FORWARD", "MOVE_BACKWARD"])
        return base_actions
    
    def get_current_address(self) -> Optional[str]:
        try:
            address_element = self.wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["address_element"])
                )
            )
            address_text = address_element.text.strip()
            address_title = address_element.get_attribute("title") or ""
            return (
                address_title
                if len(address_title) > len(address_text)
                else address_text
            )
        except Exception:
            return "Stealth Mode"

    def pan_view(self, direction: str, degrees: int = 45):
        """Pans the view using a direct JS call."""
        pov = self.driver.execute_script("return window.panorama.getPov();")
        if direction == "left":
            pov["heading"] -= degrees
        elif direction == "right":
            pov["heading"] += degrees
        self.driver.execute_script("window.panorama.setPov(arguments[0]);", pov)
        time.sleep(0.5)

    def move(self, direction: str):
        """Moves by finding the best panorama link and setting it via JS."""
        pov = self.driver.execute_script("return window.panorama.getPov();")
        links = self.driver.execute_script("return window.panorama.getLinks();")
        if not links:
            return

        current_heading = pov["heading"]
        best_link = None

        if direction == "forward":
            min_diff = 360
            for link in links:
                diff = 180 - abs(abs(link["heading"] - current_heading) - 180)
                if diff < min_diff:
                    min_diff = diff
                    best_link = link
        elif direction == "backward":
            target_heading = (current_heading + 180) % 360
            min_diff = 360
            for link in links:
                diff = 180 - abs(abs(link["heading"] - target_heading) - 180)
                if diff < min_diff:
                    min_diff = diff
                    best_link = link

        if best_link:
            self.driver.execute_script(
                "window.panorama.setPano(arguments[0]);", best_link["pano"]
            )
            time.sleep(2.5)

    def select_map_location_and_guess(self, lat: float, lon: float):
        """Minimalist guess confirmation."""
        self.driver.execute_script(
            "document.querySelector('#bottom-box').style.display = 'block';"
        )
        self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["go_button"]))
        ).click()
        time.sleep(0.5)
        self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["confirm_button"]))
        ).click()
        time.sleep(3)

    def get_ground_truth_location(self) -> Optional[Dict[str, float]]:
        """Directly gets location from JS object."""
        return self.driver.execute_script("return window.loc;")

    def click_go_button(self) -> bool:
        self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["go_button"]))
        ).click()
        time.sleep(DATA_COLLECTION_CONFIG.get("wait_after_go", 3))
        return True

    def take_street_view_screenshot(self) -> Optional[bytes]:
        pano_element = self.wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, SELECTORS["pano_container"])
            )
        )
        return pano_element.screenshot_as_png

    def load_location_from_data(self, location_data: Dict) -> bool:
        pano_id, pov = location_data.get("pano_id"), location_data.get("pov")
        if pano_id and pov:
            self.driver.execute_script(
                "window.panorama.setPano(arguments[0]); window.panorama.setPov(arguments[1]);",
                pano_id,
                pov,
            )
            time.sleep(2)
            return True
        return False

    def close(self):
        if self.driver:
            self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load_url(self, url):
        """Load a specific MapCrunch URL."""
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for the page to load
            return True
        except Exception as e:
            print(f"Error loading URL: {e}")
            return False
