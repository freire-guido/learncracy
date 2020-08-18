void setup(){
  size(1024, 768);
  rectMode(CENTER);
}

void draw(){
  background(0);
  ArrayList<float[]> arr = new ArrayList<float[]>();
  for(int i=0; i<5; i++){
    float[] nestarr = new float[5];
    for(int j=0; j<nestarr.length; j++){
      nestarr[j] = random(1,10);
    }
    arr.add(nestarr);
  }
  NLayer layer1 = new NLayer(arr.get(0), arr.get(1), arr.get(2), arr.get(3));
  layer1.display();
}

class NLayer {
  float[] inputs = new float[5];
  float[] weights = new float[5];
  float[] biases = new float[5];
  float[] outputs = new float[5];
  
  NLayer(float[] i, float[] w, float[] b, float[] o){
    inputs = i;
    weights = w;
    biases = b;
    outputs = o;
  }
  
  void forward(){
   for(int i=0; i<inputs.length; i++){
     outputs[i] = inputs[i] * weights[i] + biases[i];
   }
  }
  
  void display(){
    for(int i=0; i<outputs.length; i++){
      circle(width/2, height/2 + i*60, 50);
    }
  }
  
}
