//Create neural networks
ArrayList<NLayer>candidate0 = new ArrayList<NLayer>();
ArrayList<NLayer>candidate1 = new ArrayList<NLayer>();
ArrayList<NLayer>candidate2 = new ArrayList<NLayer>();
ArrayList<NLayer>candidate3 = new ArrayList<NLayer>();
ArrayList<ArrayList<NLayer>> population = new ArrayList<ArrayList<NLayer>>();
ArrayList<Voter> voters;
boolean newSim;
int iterations;
float[] issues;
int results;
float[] loss = {0.5};

void setup() {
  background(0);
  size(1280, 720);
  textAlign(CENTER);
  //Generate random voters
  voters = new ArrayList<Voter>();
  for (int i=0; i<5; i++) {
    float[] ran1 = {random(1), random(1), random(1)};
    voters.add(new Voter(ran1));
    if (i==0) {
      issues = ran1;
    } else {
      issues = concat(issues, ran1);
    }
  }
  //Shape neural networks
  candidate0.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate0.add(new NLayer(candidate0.get(0).outputs, 5, 2));
  candidate0.add(new NLayer(candidate0.get(1).outputs, 5, 3));
  candidate0.add(new NLayer(candidate0.get(2).outputs, voters.get(0).params.length, 4));

  candidate1.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate1.add(new NLayer(candidate1.get(0).outputs, 5, 2));
  candidate1.add(new NLayer(candidate1.get(1).outputs, 5, 3));
  candidate1.add(new NLayer(candidate1.get(2).outputs, voters.get(0).params.length, 4));

  candidate2.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate2.add(new NLayer(candidate2.get(0).outputs, 5, 2));
  candidate2.add(new NLayer(candidate2.get(1).outputs, 5, 3));
  candidate2.add(new NLayer(candidate2.get(2).outputs, voters.get(0).params.length, 4));

  candidate3.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate3.add(new NLayer(candidate3.get(0).outputs, 5, 2));
  candidate3.add(new NLayer(candidate3.get(1).outputs, 5, 3));
  candidate3.add(new NLayer(candidate3.get(2).outputs, voters.get(0).params.length, 4));

  population.add(candidate0);
  population.add(candidate1);
  population.add(candidate2);
  population.add(candidate3);
}

void draw() {
  if (newSim) {
    background(0);
    iterations++;
    //Generate a new random set of voters
    voters = new ArrayList<Voter>();
    for (int i=0; i<5; i++) {
      float[] ran1 = {random(1), random(1), random(1)};
      voters.add(new Voter(ran1));
      if (i==0) {
        issues = ran1;
      } else {
        issues = concat(issues, ran1);
      }
    }
    //Forward networks
    for (int i=0; i<population.size(); i++) {
      population.get(i).get(0).forward(issues);
      population.get(i).get(0).display(0, 0, 0);
      for (int l=1; l<population.get(i).size(); l++) {
        population.get(i).get(l).forward(population.get(i).get(l-1).outputs);
      }
    }
    //Run votes
    for (int i=0; i<voters.size(); i++) {
      voters.get(i).vote(population);
      voters.get(i).display(width*(i+1)/(voters.size()*4/3 ), -600);
    }
    //Draw networks
    for (int i=candidate0.size()-1; i>=0; i--) {
      candidate0.get(i).display(width/population.size(), 10, 0);
      candidate1.get(i).display(width/population.size(), 10, 1);
      candidate2.get(i).display(width/population.size(), 10, 2);
      candidate3.get(i).display(width/population.size(), 10, 3);
    }
    graph(results, iterations);
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
    inputs = in;
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

  void forward(float[] input) {
    //Update output if layer is the input layer
    if (pos==1) {
      outputs = input;
      //Perform vector math otherwise
    } else {
      inputs = input;
      outputs = new float[outputs.length];
      float sum = 0;
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
        sum += outputs[o];
      }
      for (int o=0; o<outputs.length; o++) {
        //Map to a value between 0 and 1
        outputs[o] = outputs[o]/sum;
      }
    }
  }

  void display(int pX, int pY, int cN) {
    int oSize = 24;
    for (int o=0; o<outputs.length; o++) {
      int oX = oSize + pos*oSize*2 + cN*pX;
      int oY = (o+1)*(oSize*10/(outputs.length+1)) + pY;   
      //Draw each synapse stroke weight = synapse weight, unless the layer is the first one
      if (pos > 1) {
        for (int i=0; i<inputs.length; i++) {
          int iX = oSize + (pos-1)*oSize*2 + cN*pX;
          int iY = (i+1)*(oSize*10/(inputs.length+1)) + pY;
          strokeWeight(abs(weights.get(o)[i]));
          line(iX, iY, oX, oY);
        }
      }
      //Draw each neuron as a circle, with its output in the middle
      strokeWeight(2);
      if (cN==0) {
        fill(0, 102, 153);
      } 
      if (cN==1) {
        fill(153, 102, 0);
      }
      if (cN==2) {
        fill(102, 0, 153);
      }
      if (cN==3) {
        fill(102, 153, 0);
      }
      stroke(255);
      circle(oX, oY, oSize);
      fill(255);
      textSize(oSize/2);
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

  void vote(ArrayList<ArrayList<NLayer>> population) {
    float[] distance = new float[population.size()];
    //Calculate distance to candidates
    for (int i=0; i<population.size(); i++) {
      ArrayList<NLayer> candidate = population.get(i);
      for (int o=0; o<candidate.get(candidate.size()-1).outputs.length; o++) {
        distance[i] += sq(candidate.get(candidate.size()-1).outputs[o] - params[o]);
      }
      distance[i] = sqrt(distance[i]);
    }
    distance[1] = sqrt(distance[1]);
    for (int i=0; i<distance.length; i++) {
      if (distance[i] == max(distance)) {
        output = i;
      }
    }
  }

  void display(int pX, int pY) {
    int oSize = 16;
    int oX = pX;
    int oY = width*3/4 - oSize/2 + pY;    
    //Draw each voter as a circle, with its vote in the middle
    strokeWeight(1.5);
    if (output==0) {
      fill(0, 102, 153);
    } 
    if (output==1) {
      fill(153, 102, 0);
    }
    if (output==2) {
      fill(102, 0, 153);
    }
    if (output==3) {
      fill(102, 153, 0);
    }
    stroke(255);
    circle(oX, oY, oSize);
    fill(255);
    textSize(14);
  }
}

void evolve(ArrayList<ArrayList<NLayer>> population) {
}

//Generates a child genome out of two parent networks
ArrayList<ArrayList<float[]>> crossOver(ArrayList<NLayer> female, ArrayList<NLayer> male) {
  ArrayList<ArrayList<float[]>> maleGenome = decode(male);
  ArrayList<ArrayList<float[]>> femGenome = decode(female);
  ArrayList<ArrayList<float[]>> childGenome = new ArrayList<ArrayList<float[]>>();
  float[] child = {};
  float[] fem = {};
  float[] man = {};
  //Turn genome into float[]
  for (int layer=0; layer<maleGenome.size(); layer++) {
    for (int list=0; list<maleGenome.get(layer).size(); list++) {
      for (int element=0; element<maleGenome.get(layer).get(list).length; element++) {
        fem = append(fem, femGenome.get(layer).get(list)[element]);
        man = append(man, maleGenome.get(layer).get(list)[element]);
      }
    }
  }
  int cutoff = int(random(fem.length));
  child = concat(subset(fem, cutoff, fem.length-cutoff), subset(man, 0, cutoff-1));
  //Turn float[] into genome[]
  for (int l=0; l<femGenome.size(); l++) {
    ArrayList<float[]> layer = new ArrayList<float[]>();
    for (int list=0; list<femGenome.get(l).size(); list++) {
      float[] outputs = {};
      for (int element=0; element<child.length; element++) {
        append(outputs, child[0]);
      }
      layer.add(outputs);
    }
    childGenome.add(layer);
  }
  return childGenome;
}

//Returns weights and biases of a network as a genome
ArrayList<ArrayList<float[]>> decode(ArrayList<NLayer> network) {
  ArrayList<ArrayList<float[]>> genomes = new ArrayList<ArrayList<float[]>>();
  for (int layer=1; layer<network.size(); layer++) {
    ArrayList<float[]> genome = new ArrayList<float[]>();
    for (int o=0; o<network.get(layer).outputs.length; o++) {
      genome.add(network.get(layer).weights.get(o));
    }
    genome.add(network.get(layer).biases);
    genomes.add(genome);
  }
  return genomes;
}

//Applies a genome to a network
void encode(ArrayList<ArrayList<float[]>> genomes, ArrayList<NLayer> network) {
  for (int layer=1; layer<network.size(); layer++) {
    for (int g=0; g<genomes.get(layer-1).size()-1; g++) {
      network.get(layer).weights.set(g, genomes.get(layer-1).get(g));
    }
    network.get(layer).biases = genomes.get(layer-1).get(genomes.size()-1);
  }
}

void graph(int pos, int total) {
  loss = append(loss, float(pos)/float(total));
  for (int i=1; i<loss.length; i++) {
    line(i+4, -loss[i-1]*10+height, i+5, -loss[i]*100+height);
  }
  stroke(255, 0, 0);
  line(0, height-50, width, height-50);
  text(loss[loss.length-1], loss.length+4, -loss[loss.length-1]*100+height);
  stroke(155);
}

void keyPressed() {
  if (key==' ') {
    newSim = true;
  }
}
