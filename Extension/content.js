console.log("Chrome Extension go");

var allClasses = [];

var allElements = document.querySelectorAll("*");

var id = 0;

for(var i = 0; i < allElements.length; i++){
	console.log(allElements[i].nodeName.toString());
	if (allElements[i].nodeName.toString() == "video"){
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
