def downloadVideo(url):
	import youtube_dl
	ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'})
	ydl_opts = {}
	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download([url])


downloadVideo("https://www.youtube.com/watch?v=fahr069-fzE")
