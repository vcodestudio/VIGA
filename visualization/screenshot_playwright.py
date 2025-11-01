from playwright.sync_api import sync_playwright
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://0.0.0.0:8000/index.html?dataset=1")
parser.add_argument("--out", default="image/out.png")
parser.add_argument("--pdf", default="pdf/out.pdf")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fullpage", default=True)
parser.add_argument("--wait", default="networkidle", help="load/domcontentloaded/networkidle/timeout(ms)")
parser.add_argument("--delay", type=int, default=0, help="渲染后额外等待毫秒(动画/字体)")
parser.add_argument("--scale", type=float, default=2.0, help="deviceScaleFactor，1~3")
args = parser.parse_args()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        viewport={"width": args.width, "height": args.height},
        device_scale_factor=args.scale
    )
    page = ctx.new_page()

    # 等待策略
    wait_until = args.wait if args.wait in ("load","domcontentloaded","networkidle") else "load"
    page.goto(args.url, wait_until=wait_until)

    if args.delay > 0:
        page.wait_for_timeout(args.delay)

    # 截图
    page.screenshot(path=args.out, full_page=args.fullpage)
    print(f"[OK] PNG: {args.out}")

    # 可选：导出 PDF（基于 Chromium）
    if args.pdf:
        # A4 纵向，也可设置 width/height，自定义页边距
        page.pdf(path=args.pdf, format="A4", margin={"top":"10mm","right":"10mm","bottom":"10mm","left":"10mm"})
        print(f"[OK] PDF: {args.pdf}")

    browser.close()
