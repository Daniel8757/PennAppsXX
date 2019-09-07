var picArray[];

function repeatCheck(){
	var mycanvas = document.createElement("CANVAS");
	var vids = document.getElementsByTagName('video');

	for(var i = 0;i<vids.length;i++){
  		vids.item(i).addEventListener('loadeddata', onLoaded);
			
		var video = vids.item(i);
		var thecanvas = mycanvas;

		video.addEventListener('pause', function(){
			draw(video, thecanvas);
		}, false);
		
	}

	function draw(video, thecanvas){
		var context = thecanvas.getContext('2d');
		context.drawImage(video, 0, 0, thecanvas.width, thecanvas.height);
		var ctx = thecanvas.getContext('2d');
		
		ctx.fillRect(50, 50, 50, 50);
		
		var img = new Image();
		img.src = thecanvas.toDataURL();

		window.picArray.push(img.src);
		
	}
	

	function onLoaded(event){
		console.log(event.target.src);
	}


	setTimeout(repeatCheck, 1000);

}

repeatCheck();
