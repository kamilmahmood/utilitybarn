import ipaddress
from collections.abc import Iterable, Mapping, MutableSet, Sequence
from itertools import chain
from typing import Optional
from urllib.parse import unquote

import regex


class StrSpan(object):
    def __init__(self, start: int, end: int, substr: str):
        super().__init__()
        self._start = start
        self._end = end
        self._substr = substr

    @property
    def start(self) -> int:
        """Get start index of span."""
        return self._start

    @property
    def end(self) -> int:
        """Get end index of span."""
        return self._end

    @property
    def substr(self) -> str:
        """Get sub string."""
        return self._substr

    def __repr__(self) -> str:
        return f"StrSpan({self.start!r}, {self.end!r}, {self.substr!r})"


class UrlSpan(StrSpan):
    def __init__(self, start: int, end: int, substr: str):
        super().__init__(start, end, substr)

    def __repr__(self) -> str:
        return f"UrlSpan({self.start!r}, {self.end!r}, {self.substr!r})"


class UrlFinder(object):
    _URL_HINT_REGEX: regex.Pattern[str] = regex.compile(
        r"""
    (
        (^|\b)
        (
            (
                # With Protocol and Authority e.g. https://test.com
                (?P<Proto>
                    [a-z][a-z0-9+.-]*
                )
                :[\/\\]{2} # Either ://, :\/, :/\ or :\\
                (?P<Authority>
                    (?P<UserInfo>
                        [\w%!$&'()*+,:;=.~-]*@
                    )?
                    (?P<Host>
                        (?P<IpLiteral>
                            \[[0-9a-f:]+\]
                        )
                        |
                        (?P<RegName>
                            [\w\u2700-\u27bf\p{Emoji}%.-]+
                        )
                    )
                )
            )
            |
            (
                # Without Protocol e.g. test.com
                (?P<NakedHost>
                    (?P<NakedIpLiteral>
                        \[[0-9a-f:]+\]
                    )
                    |
                    (?P<NakedRegName>
                        [\w\u2700-\u27bf\p{Emoji}%-]+(\.|%2e)[\w\u2700-\u27bf\p{Emoji}%.-]+
                    )
                )
            )
        )
        (?P<Port>
            :[0-9]{0,5}
        )?
        (?P<Path>
            [\\/][^\s\u0000-\u0020\u007f-\u009f?#"<>«»‘’“”|]*
        )?
        (?P<Query>
            \?[^\s\u0000-\u0020\u007f-\u009f#"<>«»‘’“”|]*
        )?
        (?P<Fragment>
            \#[^\s\u0000-\u0020\u007f-\u009f#"<>«»‘’“”|]*
        )?
    )""",
        regex.X | regex.IGNORECASE | regex.UNICODE,
    )
    _PROTO_SPLITTER: regex.Pattern[str] = regex.compile(r"[+.-]", regex.ASCII)
    _ALLOWED_HOSTNAME_CHARS: regex.Pattern[str] = regex.compile(
        r"(?!-)[A-Z\d_-]{1,63}(?<![_-])",
        regex.IGNORECASE,
    )
    _CHARS_TO_STRIP: frozenset[str] = frozenset((";", ":", ".", ","))
    _PAIR_CHARS_MAP: Mapping[str, str] = {
        ")": "(",
        "}": "{",
        "]": "[",
        ">": "<",
        "»": "«",
        "’": "‘",
        "”": "“",
    }
    # Prepare string of all opening and closing characters for regex compilation
    _PAIR_CHARS: str = "".join(
        map(regex.escape, chain(_PAIR_CHARS_MAP.keys(), _PAIR_CHARS_MAP.values())),
    )
    _PAIR_CHARS_FINDER: regex.Pattern[str] = regex.compile(
        f"[{_PAIR_CHARS}]",
        regex.UNICODE | regex.IGNORECASE,
    )

    def __init__(self, protos: Iterable[str] = ("http", "https", "noproto")):
        super().__init__()
        self._protos: frozenset[str] = frozenset(protos)

    def find(self, s: str, startidx: int = 0, baseidx: int = 0) -> Sequence[UrlSpan]:
        ret: list[UrlSpan] = []
        for match in UrlFinder._URL_HINT_REGEX.finditer(s, pos=startidx):
            if match.group("Proto") is not None:
                url = self._parse_url_with_proto(s, match, baseidx)
            else:
                url = self._parse_url_without_proto(s, match, baseidx)
            if url is not None:
                ret.append(url)
        return ret

    def _parse_url_with_proto(
        self,
        s: str,
        match: regex.Match[str],
        baseidx: int,
    ) -> Optional[UrlSpan]:
        """It is observed in malicious SMS where phisher use Click here.https://malicious.com/ or
        Click here-https://malicious.com/. In these case regex will capture here.https or
        here-https as protocol respectively, hence it is needed to strip away "here" before checking for
        supported protocols.
        """
        proto = match.group("Proto")
        # Noramlized protocol
        nproto = UrlFinder._PROTO_SPLITTER.split(proto)[-1]
        startidx, endidx = match.span()

        if nproto.lower() in self._protos:
            # Move start index to skip invalid part of protocol
            startidx += len(proto) - len(nproto)
        elif proto.lower() not in self._protos:
            return None

        if not UrlFinder._is_valid_port(match.group("Port")):
            return None

        endidx = UrlFinder._normalize_url_end(s, match, endidx)
        return UrlSpan(baseidx + startidx, baseidx + endidx, s[startidx:endidx])

    def _parse_url_without_proto(
        self,
        s: str,
        match: regex.Match[str],
        baseidx: int,
    ) -> Optional[UrlSpan]:
        if "noproto" not in self._protos:
            return None

        if not UrlFinder._is_valid_port(match.group("Port")):
            return None

        if not UrlFinder._is_valid_naked_host(s, match):
            return None

        startidx, endidx = match.span()
        endidx = UrlFinder._normalize_url_end(s, match, endidx)
        return UrlSpan(baseidx + startidx, baseidx + endidx, s[startidx:endidx])

    @staticmethod
    def _normalize_url_end(s: str, match: regex.Match[str], endidx: int) -> int:
        uristart = UrlFinder._get_first_start_idx(match, ["Path", "Query", "Fragment"])
        if uristart != -1:
            endidx = UrlFinder._strip_uri_end(s[uristart:endidx], endidx)
        elif match.group("Port") is None:
            """There is no URI and Port hence it is need to strip "." from end
            of the host e.g. https://google.com.... or google.com...."""
            endidx = UrlFinder._strip_host_end(match.group("RegName"), endidx)
        return endidx

    @staticmethod
    def _is_valid_naked_host(s: str, match: regex.Match[str]) -> bool:
        ipv6 = None
        if (host := match.group("NakedRegName")) is not None:
            hoststart, hostend = match.span("NakedRegName")
            if hoststart != 0 and s[hoststart - 1] == "@":
                # It is host part of an email address e.g. cool@domain.com
                return False
            if hostend < len(s) and s[hostend] == "@":
                # It is username part of an email address e.g. cool.com@good.com
                return False
        else:
            ipv6 = host = match.group("NakedIpLiteral")[1:-1]

        if ipv6 is None:
            try:
                # Unquote % encoding e.g. %65%78%61%6d%70%6c%65%2e%63%6f%6d -> example.com
                host = unquote(host, encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                pass

        try:
            ipaddress.ip_address(host)
        except ValueError:
            if ipv6 is not None:
                # It is an invalid IPv6
                return False
        else:
            # Either a valid IPv4 or IPv6
            return True

        try:
            # Convert unicode domain names e.g. i❤️.ws to xn--mp8hai.fm
            encoded = host.encode("idna")
        except UnicodeError:
            return False
        host = str(encoded, encoding="ascii")

        if not host or len(host) > 255:
            return False

        # TODO(@kamilmahmood): Validate TLD of host
        # Strip all dots from right e.g. google.com... to google.com
        host = host.rstrip(".")
        if not all(
            UrlFinder._ALLOWED_HOSTNAME_CHARS.fullmatch(x) for x in host.split(".")
        ):
            return False
        return True

    @staticmethod
    def _strip_uri_end(uri: str, currend: int) -> int:
        end = idx = len(uri) - 1
        stripped: MutableSet[str] = set()
        # Find occurrances of all paired chars e.g. (), {} and <> etc.
        pairchars = [
            (m.group(), m.start()) for m in UrlFinder._PAIR_CHARS_FINDER.finditer(uri)
        ]
        while idx >= 0:
            c = uri[idx]
            if c not in stripped and c in UrlFinder._CHARS_TO_STRIP:
                # Only strip a character once e.g. /path/to.. becomes /path/to.
                stripped.add(uri[idx])
                idx -= 1
            elif (opening := UrlFinder._PAIR_CHARS_MAP.get(c)) is not None:
                if UrlFinder._strip_balanced_char(pairchars, c, opening, idx):
                    idx -= 1
                else:
                    break
            else:
                break
        return currend - (end - idx)

    @staticmethod
    def _strip_host_end(host: Optional[str], currend: int) -> int:
        if host is None:
            return currend
        stripped = host.rstrip(".")
        return currend - (len(host) - len(stripped))

    @staticmethod
    def _is_valid_port(port: str) -> bool:
        if not port:
            # Empty port is a valid case e.g. https://google.com:/
            return True

        try:
            iport = int(port)
        except ValueError:
            return False
        return 0 <= iport <= 65535

    @staticmethod
    def _get_first_start_idx(match: regex.Match[str], groupnames: Sequence[str]) -> int:
        for groupname in groupnames:
            span: tuple[int, int] = match.span(groupname)
            if span[0] != -1:
                return span[0]
        return -1

    @staticmethod
    def _strip_balanced_char(
        pairchars: Sequence[tuple[str, int]],
        closing: str,
        opening: str,
        endidx: int,
    ) -> bool:
        """Check whether to strip this :closing char by counting its opening in :pairchars"""
        opened = 0
        for c, idx in pairchars:
            if idx >= endidx:
                break
            if c == opening:
                opened += 1
            elif c == closing and opened > 0:
                opened -= 1
        if opening == closing:
            """For chars that are used for both opening and closing e.g. ''.
            Return True if even number of such characters are found because all
            opened are closed hence last one is an extra and it need to be stripped
            """
            return opened % 2 == 0
        return opened == 0


__all__ = ["StrSpan", "UrlFinder", "UrlSpan"]
