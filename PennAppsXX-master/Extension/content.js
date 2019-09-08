var picArray = [];
var number = 0;
var counter = 0;


function repeatCheck(){
	window.counter = window.counter+1;
	var mycanvas = document.createElement("CANVAS");
	var vids = document.getElementsByTagName('video');

	for(var i = 0;i<vids.length;i++){
  		vids.item(i).addEventListener('loadeddata', onLoaded);
			
		var video = vids.item(i);
		var thecanvas = mycanvas;
		draw(video, thecanvas);
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


	if(window.counter<=10){
		setTimeout(repeatCheck, 1000);
	}

}

repeatCheck();

function loadDoc() {
  	var request = new XMLHttpRequest();
	request.onload = () => console.log(request.response);
	request.open("GET", "http://127.0.0.1:3000?file=yourfile");
	request.send();
}
	
loadDoc();
