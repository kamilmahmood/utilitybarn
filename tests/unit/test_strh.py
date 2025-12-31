import pytest

from utilitybarn.strh import UrlFinder


class TestUrlFinder:
    @pytest.fixture
    def httpfinder(self) -> UrlFinder:
        return UrlFinder(protos={"http", "https", "noproto"})

    @pytest.fixture
    def randfinder(self) -> UrlFinder:
        return UrlFinder(protos={"rdar", "x-yojimbo-item", "message"})

    @pytest.mark.parametrize(
        ("s", "expected"),
        [
            pytest.param("", (-1, -1), id="Empty string"),
            pytest.param(
                "https://username:password@subdomain.test.com/path;parameter/quick?q=20#fragment",
                (0, 79),
                id="URL with all possible parts",
            ),
            pytest.param(
                "abc:cool-a-http://foo.com/blah_blah",
                (11, 35),
                id="Protocol connected with hyphen",
            ),
            pytest.param("👁👄👁.fm", (0, 6), id="All emojis chars in domain name"),
            pytest.param(
                """https://i❤️.ws/emojidomain/👀🌵?format=json""",
                (0, 41),
                id="Emoji in domain name as last char and path",
            ),
            pytest.param(
                """http://foo😀.com/blah_blah/ exact location of these""",
                (0, 26),
                id="Emoji in domain name as last char",
            ),
            pytest.param(
                """(Something like http://foo.com/blah_blah)""",
                (16, 40),
                id="Written inside round brackets",
            ),
            pytest.param(
                """http://foo.com/blah_blah_(wikipedia)""",
                (0, 36),
                id="Balanced round brackets in path",
            ),
            pytest.param(
                """http://foo.com/more_(than)_one_(parens)""",
                (0, 39),
                id="Multiple balanced round brackets in path",
            ),
            pytest.param(
                """(Something like http://foo.com/blah_blah_(wikipedia))""",
                (16, 52),
                id="Written inside round brackets with balanced round brakcets in paths",
            ),
            pytest.param(
                """http://foo.com/blah_(wikipedia)#cite-1""",
                (0, 38),
                id="Balanced round brackets in path with fragment",
            ),
            pytest.param(
                """http://foo.com/blah_(wikipedia)_blah#cite-1""",
                (0, 43),
                id="Balanced round brackets in path with fragment",
            ),
            pytest.param(
                """http://foo.com/unicode_(✪)_in_parens""",
                (0, 36),
                id="Balanced round brackets in path having unicode",
            ),
            pytest.param(
                """http://foo.com/(something)?after=parens""",
                (0, 39),
                id="Balanced round brackets in path with query parameters",
            ),
            pytest.param(
                """http://foo.com/blah_blah.""",
                (0, 24),
                id="Dot at the end of path",
            ),
            pytest.param(
                """http://foo.com/blah_blah/.""",
                (0, 25),
                id="Dot at the end of path with slash",
            ),
            pytest.param(
                """<http://foo.com/blah_blah)>""",
                (1, 25),
                id="Written inside angle brackets with closing round bracket",
            ),
            pytest.param(
                """<http://foo.com/blah_blah/>""",
                (1, 26),
                id="Written inside angle brackets",
            ),
            pytest.param(
                """http://foo.com/blah_blah,""",
                (0, 24),
                id="With comma at the end",
            ),
            pytest.param(
                """http://www.extinguishedscholar.com/wpglob/?p=364""",
                (0, 48),
                id="With path and query parameter",
            ),
            pytest.param("""http://✪df.ws/1234""", (0, 18), id="First unicode char"),
            pytest.param(
                """http://➡.ws/䨹""",
                (0, 13),
                id="Single emoji char domain name with unicode in path",
            ),
            pytest.param(
                """www.c.ws/䨹""",
                (0, 10),
                id="Single char domain name with unicode in path",
            ),
            pytest.param(
                """<tag>http://example.com/</tag>""",
                (5, 24),
                id="Written inside html tag",
            ),
            pytest.param(
                """Just a www.example.com link.""",
                (7, 22),
                id="Without protocol",
            ),
            pytest.param(
                """http://example.com/something?with,commas,in,url,""",
                (0, 47),
                id="Commas in url",
            ),
            pytest.param(
                """<mailto:gruber@daringfireball.net?subject=TEST> (including brokets).""",
                (-1, -1),
                id="Mailto protocol",
            ),
            pytest.param("""mailto:name@example.com""", (-1, -1), id="Mailto protocol"),
            pytest.param(
                """bit.ly/foo""",
                (0, 10),
                id="Without protocol and with path",
            ),
            pytest.param(
                """“is.gd/foo/”""",
                (1, 11),
                id="Written inside unicode quotes",
            ),
            pytest.param(
                """WWW.EXAMPLE.COM""",
                (0, 15),
                id="All caps FQDN without protocol",
            ),
            pytest.param(
                """http://www.asianewsphoto.com/(S(neugxif4twuizg551ywh3f55))/Web_ENG/View_DetailPhoto.aspx?PicId=752""",
                (0, 98),
                id="Nested balanced round brackets in first segment of path",
            ),
            pytest.param(
                """http://www.asianewsphoto.com/(S(neugxif4twuizg551ywh3f55))""",
                (0, 58),
                id="Nested balanced round brackets in last segment of path",
            ),
            pytest.param(
                """http://lcweb2.loc.gov/cgi-bin/query/h?pp/horyd:@field(NUMBER+@band(thc+5a46634))""",
                (0, 80),
                id="Nested balanced round brackets in query parameter name",
            ),
            pytest.param(
                """http://foo.com/more_(than)_one_(parens)""",
                (0, 39),
                id="Multiple balanced round brackets",
            ),
            pytest.param(
                """{cool.com/}""",
                (1, 10),
                id="Inside curly brackets without protocol",
            ),
            pytest.param("""a.com""", (0, 5), id="Single char domain without protocol"),
            pytest.param(
                """http://user:password@cool.com/""",
                (0, 30),
                id="With user and password",
            ),
            pytest.param(
                """"https://google.com/hello""",
                (1, 25),
                id="Unbalanced double quote at the start",
            ),
            pytest.param(
                """https://asdfasdfasdf.com/lander?a=%3Cxyz.com%3E""",
                (0, 47),
                id="Percent escaped chars in query parameter value",
            ),
            pytest.param(
                """https://www.google.com?q={20}""",
                (0, 29),
                id="Query parameter value written in curl brackets",
            ),
            pytest.param(
                """{COPY}http://example.com/{}/COPY}""",
                (6, 32),
                id="Unbalanced curl brackets",
            ),
            pytest.param(
                """<a href=http://google.com/a%2f'>Click here</a>""",
                (8, 31),
                id="Written in value of html tag attribute",
            ),
            pytest.param(
                """<a href="http://google.com/a%2f/<path2>/">Click</a>""",
                (9, 32),
                id="Written in value of html tag attribute with balanced angle brackets in path",
            ),
            pytest.param(
                """(a href=http://google.com/a%2f/(path2)/)Clickhere(/a)""",
                (8, 53),
                id="Written inside unbalanced round brackets",
            ),
            pytest.param(
                """https://account.booking.comんdetailんrestric-access.www-account-booking.com/en/""",
                (0, 77),
                id="Forward slash like unicode char in domain name",
            ),
            pytest.param(
                """https:\\\\account.booking.comんdetailんrestric-access.www-account-booking.com/en/""",
                (0, 77),
                id="Reverse slashes in protocol",
            ),
        ],
    )
    def test_find_with_http_protocol(
        self,
        httpfinder: UrlFinder,
        s: str,
        expected: tuple[int, int],
    ) -> None:
        TestUrlFinder._run_test_case(httpfinder, s, expected)

    @pytest.mark.parametrize(
        ("s", "expected"),
        [
            pytest.param("""rdar://1234""", (0, 11), id="Other protocol"),
            pytest.param("""rdar:/1234""", (-1, -1), id="Invalid protocol"),
            pytest.param("""http://abc.com/""", (-1, -1), id="Http protocol"),
            pytest.param(
                """x-yojimbo-item://6303E4C1-6A6E-45A6-AB9D-3A908F59AE0E""",
                (0, 53),
                id="Protocol with hyphens",
            ),
            pytest.param(
                """message://%3c330e7f840905021726r6a4ba78dkf1fd71420c1bf6ff@mail.gmail.com%3e""",
                (0, 75),
                id="Message protocol",
            ),
        ],
    )
    def test_find_with_rand_protocol(
        self,
        randfinder: UrlFinder,
        s: str,
        expected: tuple[int, int],
    ) -> None:
        TestUrlFinder._run_test_case(randfinder, s, expected)

    @staticmethod
    def _run_test_case(urlfinder: UrlFinder, s: str, expected: tuple[int, int]) -> None:
        sidx, eidx = expected
        found = urlfinder.find(s)
        if eidx == -1:
            assert len(found) == 0, f"False url found from {s!r}"
            return
        assert len(found) == 1, f"Url not found from {s!r}"
        url = found[0]
        assert url.start == sidx, f"Invalid start index {url.start} found for {s!r}"
        assert url.end == eidx, f"Invalid end index {url.end} found for {s!r}"


__all__ = ["TestUrlFinder"]
