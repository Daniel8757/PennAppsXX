function repeatCheck(){
	var allClasses = [];
	var allElements = document.querySelectorAll("*");

	for(var i = 0; i < allElements.length; i++){
		if (allElements[i].nodeName.toString() == "video"){
			console.log(allElements[i].nodeName.toString());
			console.log("video here");
			console.log(allElements[i].src);
		}
	} 

	var vids = document.getElementsByTagName('video');

	function onLoaded(event) {
		console.log(event.target.src);
	}

	for(var i = 0;i<vids.length;i++){
		console.log(vids.item(i).src);
  		vids.item(i).addEventListener('loadeddata', onLoaded);
	}
	console.log("Chrome Extension <3");
	setTimeout(repeatCheck, 5000);

}

repeatCheck();