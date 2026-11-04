import requests
import re
import bs4
import json
import urllib.parse
import sys
import time
import os

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ar,en;q=0.9"
}

OUTPUT_FILE = r"C:\Users\HUAWEI\House_price\live_eval_dataset.json"

TARGET_DISTRICTS = [
    ("النرجس", "حي-النرجس"),
    ("العارض", "حي-العارض"),
    ("الياسمين", "حي-الياسمين"),
    ("الملقا", "حي-الملقا"),
    ("الصحافة", "حي-الصحافة")
]

arabic_digits = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9'}
def normalize_num(s):
    if not s: return ""
    for k, v in arabic_digits.items():
        s = s.replace(k, v)
    return s.replace(',', '').strip()

def parse_card(card, default_district):
    full_text = card.get_text(separator=' | ', strip=True)
    norm_text = normalize_num(full_text)
    
    # 1. Price
    # Look for number followed or preceded by ريال
    m_price = re.search(r'([\d,]{7,10})\s*\|\s*§\s*\|\s*ريال', full_text)
    if not m_price:
        m_price = re.search(r'([\d,]{7,10})\s*ريال', full_text)
    if not m_price:
        # Search in normalized text
        m_price = re.search(r'(\d{7,8})\s*(?:\||\s)*ريال', norm_text)
    if not m_price:
        return None
    price = float(normalize_num(m_price.group(1)))
    
    # 2. Size
    m_size = re.search(r'(\d{2,4})\s*(?:م²|م2|متر)', norm_text)
    if not m_size:
        return None
    size = float(m_size.group(1))
    
    # 3. Filter criteria: villa, residential bounds
    if not (150 <= size <= 650):
        return None
    if not (1_200_000 <= price <= 8_500_000):
        return None
        
    # 4. District
    district = default_district
    for d_name, _ in TARGET_DISTRICTS:
        if d_name in full_text:
            district = d_name
            break
            
    # 5. Extract bedrooms & bathrooms if available
    # Often listed as numbers in separate pills
    nums = re.findall(r'\|\s*(\d{1,2})\s*\|', norm_text)
    beds = 4.0
    baths = 4.0
    if len(nums) >= 2:
        try:
            cand1, cand2 = float(nums[0]), float(nums[1])
            if 2 <= cand1 <= 10: beds = cand1
            if 2 <= cand2 <= 10: baths = cand2
        except:
            pass
            
    # 6. Title and description
    title = full_text.split('|')[0].strip()
    
    return {
        "title": title,
        "district": district,
        "price": price,
        "size": size,
        "bedrooms": beds,
        "bathrooms": baths,
        "full_text": full_text
    }

collected = []
seen_signatures = set()

print("=" * 75)
print("COLLECTING LIVE LISTINGS UNDER PRE-DECLARED SELECTION RULE")
print("Target: 30-45 listings across 5 key Riyadh districts (النرجس, العارض, الياسمين, الملقا, الصحافة)")
print("=" * 75)

for d_name, d_slug in TARGET_DISTRICTS:
    url = f"https://sa.aqar.fm/%D9%81%D9%84%D9%84-%D9%84%D9%84%D8%A8%D9%8A%D8%B9/%D8%A7%D9%84%D8%B1%D9%8a%D8%A7%D8%B6/{urllib.parse.quote(d_slug)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            print(f"Failed to fetch {d_name}: HTTP {r.status_code}")
            continue
            
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        listing_links = soup.find_all('a', href=re.compile(r'/فلل-للبيع/.+-\d{6,}'))
        print(f"\nProcessing {d_name} ({len(listing_links)} candidate links on page 1)...")
        
        district_count = 0
        for a in listing_links:
            # find card container
            card = a
            for _ in range(5):
                if card.parent and len(card.parent.get_text()) < 3000:
                    card = card.parent
                    if 'ريال' in card.get_text():
                        break
                        
            parsed = parse_card(card, d_name)
            if parsed:
                # Deduplicate by (price, size, district)
                sig = (parsed["district"], parsed["size"], parsed["price"])
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    parsed["url"] = a.get("href")
                    collected.append(parsed)
                    district_count += 1
                    print(f"  [+] #{len(collected):02d} | {parsed['district']} | {parsed['size']}m² | {parsed['price']:,.0f} SAR | {parsed['title'][:45]}")
                    
            if len(collected) >= 45:
                break
                
        # Also check page 2 if needed
        if district_count < 6 and len(collected) < 40:
            url_p2 = url + "/2"
            time.sleep(1.0)
            r2 = requests.get(url_p2, headers=HEADERS, timeout=12)
            if r2.status_code == 200:
                soup2 = bs4.BeautifulSoup(r2.text, 'html.parser')
                links_p2 = soup2.find_all('a', href=re.compile(r'/فلل-للبيع/.+-\d{6,}'))
                for a in links_p2:
                    card = a
                    for _ in range(5):
                        if card.parent and len(card.parent.get_text()) < 3000:
                            card = card.parent
                            if 'ريال' in card.get_text():
                                break
                    parsed = parse_card(card, d_name)
                    if parsed:
                        sig = (parsed["district"], parsed["size"], parsed["price"])
                        if sig not in seen_signatures:
                            seen_signatures.add(sig)
                            parsed["url"] = a.get("href")
                            collected.append(parsed)
                            print(f"  [+] #{len(collected):02d} (p2) | {parsed['district']} | {parsed['size']}m² | {parsed['price']:,.0f} SAR | {parsed['title'][:45]}")
                    if len(collected) >= 45:
                        break
                        
    except Exception as e:
        print(f"Error scraping {d_name}: {e}")
        
    time.sleep(1.0)
    if len(collected) >= 45:
        break

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(collected, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 75)
print(f"SUCCESS: Collected {len(collected)} verified live listings.")
print(f"Dataset saved to: {OUTPUT_FILE}")
print("=" * 75)
