from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

class StartPage:

    def __init__(self, driver, url):
        self.driver = driver
        self.base_url = url

    def look_for_one_element(self, locator,time=10):
        return WebDriverWait(self.driver,time).until(ec.presence_of_element_located(locator),
                                                      message=f"Can't find element by locator {locator}")

    def look_for_elements(self, locator,time=10):
        return WebDriverWait(self.driver,time).until(ec.presence_of_element_located(locator),
                                                      message=f"Can't find elements by locator {locator}")

    def find_el(self, locator):
        return self.driver.find_element_by_css_selector(locator[1])

    def go_to(self):
        return self.driver.get(self.base_url)