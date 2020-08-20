//Create neural networks
ArrayList<NLayer>candidate1 = new ArrayList<NLayer>();
ArrayList<NLayer>candidate2 = new ArrayList<NLayer>();
ArrayList<Voter> voters;
boolean newSim;
float[] issues;

void setup() {
  background(0);
  size(1024, 768);
  textAlign(CENTER);
  //Generate random voters
  voters = new ArrayList<Voter>();
  for (int i=0; i<3; i++) {
    float[] ran1 = {random(1), random(1)};
    voters.add(new Voter(ran1));
    if (i==0) {
      issues = ran1;
    } else {
      issues = concat(issues, ran1);
    }
  }
  candidate1.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate1.add(new NLayer(candidate1.get(0).outputs, 6, 2));
  candidate1.add(new NLayer(candidate1.get(1).outputs, 6, 3));
  candidate1.add(new NLayer(candidate1.get(2).outputs, voters.get(0).params.length, 4));
  candidate2.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate2.add(new NLayer(candidate1.get(0).outputs, 6, 2));
  candidate2.add(new NLayer(candidate1.get(1).outputs, 6, 3));
  candidate2.add(new NLayer(candidate1.get(2).outputs, voters.get(0).params.length, 4));
}

void draw() {
  int tally = 0;
  float[] target = {20, 20};
  if (newSim) {
    background(0);
    voters = new ArrayList<Voter>();
    for (int i=0; i<5; i++) {
      float[] ran1 = {random(1), random(1), random(1)};
      voters.add(new Voter(ran1));
      if (i==0) {
        issues = ran1;
      } else {
        issues = concat(issues, ran1);
      }
      println("Generating voter:", i+1);
    }
    for (int i=0; i<candidate1.size(); i++) {
      if (i>0) {
        candidate1.get(i).forward();
        candidate2.get(i).forward();
      }
      //Draw networks
      candidate1.get(i).display(0, 0, 0);
      candidate2.get(i).display(width/2, 0, 1);
    }
    for (int i=0; i<voters.size(); i++) {
      //Draw voters
      voters.get(i).vote(candidate1.get(3).outputs, candidate2.get(3).outputs);
      tally += voters.get(i).output;
      voters.get(i).display(width/(voters.size()+1)*(i+1), -10);
    }
    //Display winner
    if (tally > voters.size()/2) {
      textSize(20);
      text("Winner: candidate2", width/2, height*2/3);
    } else {
      textSize(20);
      text("Winner: candidate1", width/2, height*2/3);
    }
    backPropagate(candidate1, target, 4);
    newSim = false;
  }
}

class NLayer {
  ArrayList<float[]> weights;
  float[] biases;  
  float[] inputs;  //Past layer's neuronal values
  float[] outputs; //Current layer's neuronal values
  int pos;
  NLayer(float[] in, int out, int p) {
    pos = p;    
    inputs = issues;
    outputs = new float[out];
    biases = new float[out];
    weights = new ArrayList<float[]>(in.length);
    //Initialize random weights and biases for neuron n, unless the layer is the first
    if (pos>1) {
      for (int o=0; o<outputs.length; o++) {
        float[] weight = new float[inputs.length];
        for (int i=0; i<inputs.length; i++) {
          weight[i] = random(1);
        }
        weights.add(weight);
        biases[o] = random(1);
      }
    } else {
      outputs = inputs;
    }
  }

  void forward() {
    outputs = new float[outputs.length];
    for (int o=0; o<outputs.length; o++) {
      for (int i=0; i<inputs.length; i++) {
        //Rectify inputs (ReLU)
        if (inputs[i]<0) {
          inputs[i] = 0;
        }
        //Multiply inputs by their corresponding weights and add the biases
        outputs[o] += inputs[i] * weights.get(o)[i];
      }
      outputs[o] += biases[o];
    }
  }

  void display(int pX, int pY, int cN) {
    int oSize = 32;
    for (int o=0; o<outputs.length; o++) {
      int oX = oSize + pos*oSize*2 + pX;
      int oY = (o+1)*(400/(outputs.length+1)) + pY;   
      //Draw each synapse stroke weight = synapse weight, unless the layer is the first one
      if (pos > 1) {
        for (int i=0; i<inputs.length; i++) {
          int iX = oSize + (pos-1)*oSize*2 + pX;
          int iY = (i+1)*(400/(inputs.length+1)) + pY;
          strokeWeight(weights.get(o)[i]);
          line(iX, iY, oX, oY);
        }
      }
      //Draw each neuron as a circle, with its output in the middle
      strokeWeight(2);
      if (cN==0) {
        fill(0, 102, 153);
      } else {
        fill(153, 102, 0);
      }
      stroke(255);
      circle(oX, oY, oSize);
      fill(255);
      textSize(10);
      text(outputs[o], oX, oY+4);
    }
  }
}

class Voter {
  ArrayList<float[]> inputs = new ArrayList<float[]>();
  int output;
  float[] params;
  Voter(float[] parameters) {
    params = parameters;
  }

  void vote(float[] input1, float[] input2) {
    float[] distance = new float[2];
    inputs.add(input1);
    inputs.add(input2);
    //Calculate distance to input1
    for (int i=0; i<input1.length; i++) {
      distance[0] += sq(params[i] - input1[i]);
    }
    distance[0] = sqrt(distance[0]);
    //Calculate distance to input2
    for (int i=0; i<input2.length; i++) {
      distance[1] += sq(params[i] - input2[i]);
    }
    distance[1] = sqrt(distance[1]);
    if (distance[1] > distance[0]) {
      output = 0;
    }
    if (distance[0] > distance[1]) {
      output = 1;
    }
  }

  void display(int pX, int pY) {
    int oSize = 36;
    int oX = pX;
    int oY = width*3/4 - oSize/2 + pY;    
    //Draw each voter as a circle, with its vote in the middle
    strokeWeight(2);
    if (output==0) {
      fill(0, 102, 153);
    } else {
      fill(153, 102, 0);
    }
    stroke(255);
    circle(oX, oY, oSize);
    fill(255);
    textSize(14);
  }
}

void back(ArrayList<NLayer> network, int n, int size) {
  int layer = size-1;
  //Calculate MSE (weights)
  print("LAY", layer, "NEU", n);
  if (layer>0) {
    int wSize = network.get(layer).weights.get(n).length;
    float MSE = 0;
    for (int w=0; w<wSize; w++) {
      MSE += network.get(layer).weights.get(n)[w];
    }
    MSE = MSE/wSize;
    for (int w=0; w<wSize; w++) {
      if (network.get(layer).weights.get(n)[w] > MSE) {
        println("        layer:", layer, "neuron:", n, "weight:", w);
        back(network, w, layer);
      }
    }
  }
}

void backPropagate(ArrayList<NLayer> network, float[] t, int size) {
  NLayer layer = network.get(size-1);
  for (int n=0; n<layer.outputs.length; n++) {
    if (layer.outputs[n] < t[n]) {
      println("NEW OUTPUT");
      //Calculate MSE (weights)
      float MSE = 0;
      for (int w=0; w<layer.weights.get(n).length; w++) {
        MSE += layer.weights.get(n)[w];
      }
      MSE = MSE/layer.weights.get(n).length;
      //println("TARGETTING OUTPUT", n);
      for (int w=0; w<layer.weights.get(n).length; w++) {
        if (layer.weights.get(n)[w] > MSE) {
          back(network, w, size-1);
        }
      }
    }
  }
}

void keyPressed() {
  if (key==' ') {
    newSim = true;
  }
}
