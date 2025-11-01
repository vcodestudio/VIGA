# shot_wait_playwright.py
from playwright.sync_api import sync_playwright
import argparse, time

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--out", default="page.png")
parser.add_argument("--pdf", default="")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fullpage", action="store_true")
parser.add_argument("--scale", type=float, default=2.0)  # HiDPI，图更清晰
parser.add_argument("--timeout", type=int, default=30000) # ms，最大等待时间
parser.add_argument("--dataset", type=int, default=1)     # 选择 ?dataset=1/2/3
args = parser.parse_args()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        viewport={"width": args.width, "height": args.height},
        device_scale_factor=args.scale,
    )
    page = ctx.new_page()
    url = args.url
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}dataset={args.dataset}"

    # 进入页面并等待网络空闲
    page.goto(url, wait_until="networkidle", timeout=args.timeout)

    # 等待“渲染完成”信号：list 有子元素或 fallback JSON 出现
    page.wait_for_function("""
        () => {
          const list = document.querySelector('#generator_list');
          const fallback = document.querySelector('#generator_json');
          const listReady = list && list.children && list.children.length > 0;
          const fallbackReady = fallback && fallback.style.display !== 'none' &&
                                fallback.textContent.trim() !== '';
          return listReady || fallbackReady || document.documentElement.dataset.ready === '1';
        }
    """, timeout=args.timeout)

    # 再等所有图片加载完
    page.evaluate("""
        () => Promise.all(
          Array.from(document.images).map(img => img.complete ? null : new Promise(r => {
            img.onload = img.onerror = () => r(null);
          }))
        )
    """)

    # 截图
    page.screenshot(path=args.out, full_page=args.fullpage)
    print(f"[OK] PNG: {args.out}")

    if args.pdf:
        page.pdf(path=args.pdf, format="A4", margin={"top":"10mm","right":"10mm","bottom":"10mm","left":"10mm"})
        print(f"[OK] PDF: {args.pdf}")

    browser.close()
