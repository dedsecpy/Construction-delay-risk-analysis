import fitz

doc = fitz.open("main.pdf")
print("pages", doc.page_count)
for i in range(min(12, doc.page_count)):
    page = doc[i]
    text = sum(len(b[4]) for b in page.get_text("blocks") if b[6] == 0)
    print(f"p{i+1}: imgs={len(page.get_images())} text={text}")
