from YouTubePages import SearchStep
from selenium import webdriver
from behave import *

@given('website "{url}"')
def step(context, url):
    context.browser = webdriver.Chrome(executable_path="./chromedriver")
    context.browser.maximize_window()
    context.browser = SearchStep(context.browser, url)
    context.browser.go_to()
    assert context.browser.driver.title == "YouTube"


@then("search '{text}'")
def search(context, text):
    context.browser.enter_word(text)


@when("push search button and it not '{text}'")
def step(context, text):
    context.browser.click_on_the_search_button()
    assert text not in context.browser.driver.page_source

@when("click first video with '{url}'")
def step(context, url):
    context.browser.check_results()
    assert context.browser.driver.current_url == url


@then("page include text '{text}'")
def step(context, text):
    elements = context.browser.find_name()
    assert elements == text