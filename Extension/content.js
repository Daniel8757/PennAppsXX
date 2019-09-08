var picArray = [];
var number = 0;

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
		
		var img = new Image();
		img.src = thecanvas.toDataURL();

		window.picArray.push(img.src);

		var myJSON = JSON.stringify(img.src);

		let dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(myJSON);

    		let exportFileDefaultName = window.number.toString();
		window.number = window.number+1;

   		let linkElement = document.createElement('a');
   		linkElement.setAttribute('href', dataUri);
    		linkElement.setAttribute('download', exportFileDefaultName);
   		linkElement.click();
		window.picArray = [];
		
	}
	

	function onLoaded(event){
		console.log(event.target.src);
	}

	setTimeout(repeatCheck, 1000);

}

repeatCheck();
