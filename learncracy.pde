ArrayList<NLayer> candidate1;
ArrayList<NLayer> candidate2;
ArrayList<Voter> voters;
boolean newSim;

void setup() {
  size(1024, 768);
  textAlign(CENTER);
  
  //Generate a random input array
  voters = new ArrayList<Voter>();
  for(int i=0; i<20; i++){
    float[] ran1 = {random(10), random(10), random(10)};
    voters.add(new Voter(ran1));
  }  
}

void draw() {
  int tally = 0;
  if(newSim){
    background(0);
    //Create neural networks
    float[] ran = {random(1), random(1), random(1), random(1)};
    candidate1 = new ArrayList<NLayer>();
    candidate2 = new ArrayList<NLayer>();
    candidate1.add(new NLayer(ran, 4, 1));
    candidate1.add(new NLayer(candidate1.get(0).outputs, 8, 2));
    candidate1.add(new NLayer(candidate1.get(1).outputs, 8, 3));
    candidate1.add(new NLayer(candidate1.get(2).outputs, 3, 4));
    candidate2.add(new NLayer(ran, 4, 1));
    candidate2.add(new NLayer(candidate1.get(0).outputs, 8, 2));
    candidate2.add(new NLayer(candidate1.get(1).outputs, 8, 3));
    candidate2.add(new NLayer(candidate1.get(2).outputs, 3, 4));
    for(int i=0; i<candidate1.size(); i++){
      candidate1.get(i).forward();
      candidate2.get(i).forward();
      candidate1.get(i).display(width/2, 0, 0);
      candidate2.get(i).display(0, 0, 1);
    }
    for(int i=0; i<voters.size(); i++){
      voters.get(i).vote(candidate1.get(3).outputs, candidate2.get(3).outputs);
      tally += voters.get(i).output;
      voters.get(i).display(width/(voters.size()+1)*(i+1), -10);
    }
    if(tally > voters.size()/2){
      textSize(20);
      text("Winner: candidate1", width/2, height*2/3);
    }
    else{
      textSize(20);
      text("Winner: candidate2", width/2, height*2/3);
    }
   newSim = false; 
  }
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

  void display(int pX, int pY, int cN) {
    int oSize = 42;
    for (int o=0; o<outputs.length; o++) {
      int oX = oSize + pos*oSize*2 + pX;
      int oY = (o+1)*(400/(outputs.length+1)) + pY;   
      //Draw each synapse stroke weight = synapse weight, unless the layer is the first one
      if(pos > 1){
        for (int i=0; i<inputs.length; i++) {
          int iX = oSize + (pos-1)*oSize*2 + pX;
          int iY = (i+1)*(400/(inputs.length+1)) + pY;
          strokeWeight(weights.get(o)[i]);
          line(iX, iY, oX, oY);
        }
      }
      //Draw each neuron as a circle, with its output in the middle
      strokeWeight(2);
      if(cN==0){
        fill(0, 102, 153); 
      }
      else{
        fill(153, 102, 0);
      }
      stroke(255);
      circle(oX, oY, oSize);
      fill(255);
      textSize(14);
      text(outputs[o], oX, oY+4);
    }
  }
}

class Voter {
  ArrayList<float[]> inputs = new ArrayList<float[]>();
  int output;
  float[] params;
  Voter(float[] parameters){
    params = parameters;
  }
  
  void vote(float[] input1, float[] input2){
    float[] distance = new float[2];
    inputs.add(input1);
    inputs.add(input2);
    //Calculate distance to input1
    for(int i=0; i<input1.length; i++){
      distance[0] += sq(params[i] - input1[i]);
    }
    distance[0] = sqrt(distance[0]);
    //Calculate distance to input2
    for(int i=0; i<input2.length; i++){
      distance[1] += sq(params[i] - input2[i]);
    }
    distance[1] = sqrt(distance[1]);
    if(distance[1] > distance[0]){
      output = 0;
    }
    if(distance[0] > distance[1]){
      output = 1;
    }
  }
  
  void display(int pX, int pY){
    int oSize = 36;
    int oX = pX;
    int oY = width*3/4 - oSize/2 + pY;    
    //Draw each voter as a circle, with its vote in the middle
    strokeWeight(2);
    if(output==0){
      fill(0, 102, 153);
    }
    else{
      fill(153, 102, 0);
    }
    stroke(255);
    circle(oX, oY, oSize);
    fill(255);
    textSize(14);
  }
}

void keyPressed(){
  if(key==' '){
    newSim = true;
  }
}
