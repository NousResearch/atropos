// blazy code
var bLazy = new Blazy({
  success: function(){
    updateCounter();
  }
});

// not needed, only here to illustrate amount of loaded images
var bImageLoaded = 0;

function updateCounter() {
  bImageLoaded++;
  console.log("blazy image loaded: "+bImageLoaded);
}