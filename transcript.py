import whisper
model = whisper.load_model("base")
result = model.transcribe("Baby(PagalNew.Com.Se).mp3")
with open("babedemo.txt","w") as f:
    f.write(result["text"])
