from MainProc import StartPage
from selenium.webdriver.common.by import By
import time


class YoutubeSearchLocators:
    LOCATOR_YOUTUBE_SEARCH_FIELD = (By.XPATH, "//input[@id=\"search\"]")
    LOCATOR_YOUTUBE_SEARCH_BUTTON = (By.XPATH, "//button[@id=\"search-icon-legacy\"]")
    LOCATOR_YOUTUBE_FIRST_RESULT = (By.XPATH, "//div[@id=\"contents\"]/ytd-item-section-renderer/div/"+
                                     "ytd-video-renderer[1]/div/ytd-thumbnail/a/yt-img-shadow/img")
    LOCATOR_YOUTUBE_VIDEO_NAME = (By.CSS_SELECTOR, "h1.title > yt-formatted-string")
    LOCATOR_YOUTUBE_VIDEO = (By.CSS_SELECTOR, "#img")

class SearchStep(StartPage):

    def enter_word(self, word):
        search_field = self.look_for_one_element(YoutubeSearchLocators.LOCATOR_YOUTUBE_SEARCH_FIELD)
        search_field.click()
        search_field.send_keys(word)
        return search_field

    def click_on_the_search_button(self):
        return self.look_for_one_element(YoutubeSearchLocators.LOCATOR_YOUTUBE_SEARCH_BUTTON,time=15).click()

    def check_results(self):
        return self.look_for_one_element(YoutubeSearchLocators.LOCATOR_YOUTUBE_FIRST_RESULT,time=20).click()

    def find_name(self):
        time.sleep(30)
        el = self.look_for_elements(YoutubeSearchLocators.LOCATOR_YOUTUBE_VIDEO_NAME, time=10)
        return el.text