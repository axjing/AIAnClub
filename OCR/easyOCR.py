import easyocr
reader = easyocr.Reader(['ch_sim','en'], download_enabled=False) # this needs to run only once to load the model into memory
result = reader.readtext(r'C:\Users\axjing\Pictures\Camera Roll\code.png')
print(result)