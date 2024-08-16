import bs4
import chardet
import requests
import tqdm


def get_bs_res(url: str) -> bs4.BeautifulSoup:
    res = requests.get(url)
    encoding = chardet.detect(res.content)["encoding"]
    bs_res = bs4.BeautifulSoup(res.content.decode(encoding), "html.parser")
    return bs_res


def get_aozora_text(url: str) -> str:
    bs_res = get_bs_res(url)
    main_text = bs_res.find("div", class_="main_text")
    tags_to_delete = main_text.find_all(["rp", "rt"])
    for tag in tags_to_delete:
        tag.decompose()
    no_tag_main_text = main_text.get_text()
    return no_tag_main_text


def get_all_natsume_soseki_text(verbose: bool = False) -> str:
    base_url = "https://www.aozora.gr.jp"
    author_url = f"{base_url}/index_pages/person148.html"
    bs_res = get_bs_res(author_url)
    all_card_url_tags = bs_res.find_all("a")
    card_url_tags = list(card_url_tag for card_url_tag in all_card_url_tags if card_url_tag.get("href") is not None)
    card_url_tags = list(filter(lambda x: x.get("href").startswith("../cards"), card_url_tags))
    if verbose:
        print(f"Found {len(card_url_tags)} candidates.")
    all_texts = {}  # dict of {book_name: text}

    for card_url_tag in tqdm.tqdm(card_url_tags):
        # カードを取得
        card_url = card_url_tag.get("href")
        card = get_bs_res(f"{base_url}{card_url[2:]}")
        book_name = card.find("meta", attrs={"property": "og:title"}).get("content")

        # 文字遣い種別が新字新仮名でない場合はスキップ
        book_table = card.find("table", summary="作品データ")
        card_is_new_kana = True
        for tr in book_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) != 2:
                raise ValueError("tds length is not 2")
            else:
                if tds[0].get_text() == "文字遣い種別：":
                    if tds[1].get_text() != "新字新仮名":
                        card_is_new_kana = False
                        break
        if not card_is_new_kana:
            if verbose:
                tqdm.tqdm.write(f"{"book_name"}: 文字遣い種別が新字新仮名でないためスキップしました。")
            continue

        card_a_tags = card.find_all("a")
        for a_tag in card_a_tags:
            a_tag_content = a_tag.contents[0]
            if a_tag_content == "いますぐXHTML版で読む" or a_tag_content == "いますぐHTML版で読む":
                text_url_surfix = a_tag.get("href")
                text_url = f"{base_url}/{'/'.join(card_url.split('/')[:-1])}{text_url_surfix[1:]}"
                text = get_aozora_text(text_url)
                all_texts[book_name] = text
                if verbose:
                    tqdm.tqdm.write(f"{book_name}: 取得しました。")
                break
    return all_texts
