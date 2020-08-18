void setup() {
  size(1024, 768);
  textAlign(CENTER);
}

void draw() {
  background(0); 
  //Generate a random input array
  float[] ran = {random(1), random(1), random(1), random(1)};
  //Define neural network shape
  NLayer layer1 = new NLayer(ran, 4, 1);
  NLayer layer2 = new NLayer(layer1.outputs, 8, 2);
  NLayer layer3 = new NLayer(layer2.outputs, 8, 3);
  NLayer layer4 = new NLayer(layer3.outputs, 2, 4);
  
  layer1.forward();
  layer2.forward();
  layer3.forward();
  layer4.forward();
  layer1.display();
  layer2.display();
  layer3.display();
  layer4.display();
}

class NLayer {
  ArrayList<float[]> weights;
  float[] biases;  
  float[] inputs;
  float[] outputs;
  int pos;
  NLayer(float[] in, int out, int p) {
    pos = p;    
    inputs = in;
    outputs = new float[out];
    biases = new float[out];
    weights = new ArrayList<float[]>(in.length);
    //Initialize random weights and biases for neuron n
    for (int o=0; o<outputs.length; o++) {
      float[] weight = new float[inputs.length];
      for (int i=0; i<inputs.length; i++) {
        weight[i] = random(1);
      }
      weights.add(weight);
      biases[o] = random(5);
    }
  }

  void forward() {
    for (int o=0; o<outputs.length; o++) {
      for (int i=0; i<inputs.length; i++) {
        //Rectify inputs (ReLU)
        if(inputs[i]<0){
          inputs[i] = 0;
        }
        //Multiply inputs by their corresponding weights and add the biases
        outputs[o] = inputs[i] * weights.get(o)[i] + biases[o];
      }
    }
  }

  void display() {
    int nSize = 50;
    for (int o=0; o<outputs.length; o++) {
      int nX = width/8 + pos*nSize*3;
      int nY = (o+1)*(height/(outputs.length+1));      
      //Draw each synapse stroke weight = synapse weight, unless the layer is the first one
      if(pos > 1){
        for (int i=0; i<inputs.length; i++) {
          int iX = width/8 + (pos-1)*nSize*3;
          int iY = (i+1)*(height/(inputs.length+1));
          strokeWeight(weights.get(o)[i]);
          line(iX, iY, nX, nY);
        }
      }
      //Draw each neuron as a circle, with its output in the middle
      strokeWeight(2);
      fill(0, 102, 153);
      stroke(255);
      circle(nX, nY, nSize);
      fill(255);
      text(outputs[o], nX, nY+4);
    }
  }
}
