Feature: Checking YouTube

  Scenario Outline: Сheck right opening of some songs
    Given website <url>
    Then search <songs name>
    When push search button and it not 'No results found'
    When click first video with <songs Url>
    Then page include text <full name>

    Examples: Possible Entries
      | url                      | songs name |songs Url                                    | full name                                     |
      |"https://www.youtube.com/"| 'rickroll' |'https://www.youtube.com/watch?v=dQw4w9WgXcQ'|'Rick Astley - Never Gonna Give You Up (Video)'|
