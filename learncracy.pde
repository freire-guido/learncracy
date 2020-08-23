//Create neural networks
ArrayList<NLayer> candidate0 = new ArrayList<NLayer>();
ArrayList<NLayer> candidate1 = new ArrayList<NLayer>();
ArrayList<NLayer> candidate2 = new ArrayList<NLayer>();
ArrayList<NLayer> candidate3 = new ArrayList<NLayer>();
ArrayList<ArrayList<NLayer>> population = new ArrayList<ArrayList<NLayer>>();
ArrayList<Voter> voters;
float[] fitness = {};
float fitsum = 0;
boolean newSim;
int iterations;
float[] issues;
int[] tally;

void setup() {
  background(0);
  size(1280, 720);
  textAlign(CENTER);
  //Generate random voters
  voters = new ArrayList<Voter>();
  for (int i=0; i<10; i++) {
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
  candidate0.add(new NLayer(candidate0.get(0).outputs, 6, 2));
  candidate0.add(new NLayer(candidate0.get(1).outputs, 6, 3));
  candidate0.add(new NLayer(candidate0.get(2).outputs, voters.get(0).params.length, 4));

  candidate1.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate1.add(new NLayer(candidate1.get(0).outputs, 6, 2));
  candidate1.add(new NLayer(candidate1.get(1).outputs, 6, 3));
  candidate1.add(new NLayer(candidate1.get(2).outputs, voters.get(0).params.length, 4));

  candidate2.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate2.add(new NLayer(candidate2.get(0).outputs, 6, 2));
  candidate2.add(new NLayer(candidate2.get(1).outputs, 6, 3));
  candidate2.add(new NLayer(candidate2.get(2).outputs, voters.get(0).params.length, 4));

  candidate3.add(new NLayer(issues, voters.size()*voters.get(0).params.length, 1));
  candidate3.add(new NLayer(candidate3.get(0).outputs, 6, 2));
  candidate3.add(new NLayer(candidate3.get(1).outputs, 6, 3));
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
    for (int i=0; i<10; i++) {
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
      for (int l=1; l<population.get(i).size(); l++) {
        population.get(i).get(l).forward(population.get(i).get(l-1).outputs);
      }
    }
    //Run and count votes
    tally = new int[population.size()];
    for (int i=0; i<voters.size(); i++) {
      voters.get(i).vote(population);
      voters.get(i).display(width*(i+1)/(voters.size()+1), -600);
      tally[voters.get(i).output]++;
    }
    printArray(tally);
    println();
    //Draw networks
    for (int c=0; c<population.size(); c++) {
      for (int i=candidate0.size()-1; i>=0; i--) {
        population.get(c).get(i).display(width/population.size(), 10, c);
      }
    }
    //Graph fitness
    fitsum += max(tally)*100/voters.size();
    //graph(max(tally)*100/voters.size(), 1);
    graph(fitsum, iterations);

    //Generate new population
    evolve(population, tally);
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
      //Map last layer's outputs to a value between 0 and 1
      if (pos==4) {
        for (int o=0; o<outputs.length; o++) {
          outputs[o] = outputs[o]/sum;
        }
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

void evolve(ArrayList<ArrayList<NLayer>> population, int[] tally) {
  ArrayList<NLayer> female = new ArrayList<NLayer>();
  ArrayList<NLayer> male = new ArrayList<NLayer>();
  int[] losers = {};
  int temp = tally[0];
  for (int i=0; i<tally.length; i++) {
    //Pick best network as a male network
    if (tally[i]==max(tally)) {
      male = population.get(i);
    } else if (tally[i]>temp && tally[i]<max(tally)) {
      female = population.get(i);
    } else {
      losers = append(losers, i);
    }
    temp = tally[i];
  }
  //If there is no runner up, female = male  
  if (female.size()==0) {
    female = male;
  }
  //Make losing networks child of winner networks, with a small chance of mutating
  for (int i=0; i<losers.length; i++) {
    float dice = random(1); 
    encode(crossOver(female, male), population.get(losers[i]));
    if (dice<0.1) {
      encode(mutate(population.get(losers[i])), population.get(i));
    }
  }
}

//Mutates a network (changes random chromosomes to random values)
ArrayList<ArrayList<float[]>> mutate(ArrayList<NLayer> network) {
  //println("Mutating");
  ArrayList<ArrayList<float[]>> netGenome = decode(network);
  ArrayList<ArrayList<float[]>> mutatedGenome = decode(network);
  for (int layer=0; layer<netGenome.size(); layer++) {
    for (int list=0; list<netGenome.get(layer).size(); list++) {
      for (int element=0; element<netGenome.get(layer).get(list).length; element++) {
        float dice = random(1);
        if (dice<0.1) {
          float mutation = random(1);
          netGenome.get(layer).get(list)[element] = mutation;
        }
      }
    }
  }
  return mutatedGenome;
}

//Generates a child genome out of two parent networks
ArrayList<ArrayList<float[]>> crossOver(ArrayList<NLayer> female, ArrayList<NLayer> male) {
  //println("Crossing");
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
  int cutoff = int(random(2, fem.length));
  child = concat(subset(fem, cutoff, fem.length-cutoff), subset(man, 0, cutoff));
  //Turn float[] into genome[]
  int prevLength = 0;
  for (int layer=0; layer<femGenome.size(); layer++) {
    ArrayList<float[]> l = new ArrayList<float[]>();
    for (int list=0; list<femGenome.get(layer).size(); list++) {
      float[] outputs = {};
      for (int element=0; element<femGenome.get(layer).get(list).length; element++) {
        outputs = append(outputs, child[element+prevLength]);
      }
      prevLength += femGenome.get(layer).get(list).length;
      l.add(outputs);
    }
    childGenome.add(l);
  }
  return childGenome;
}

//Returns weights and biases of a network as a genome
ArrayList<ArrayList<float[]>> decode(ArrayList<NLayer> network) {
  //println("Decoding");
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
  //println("Encoding");
  for (int layer=1; layer<network.size(); layer++) {
    for (int g=0; g<genomes.get(layer-1).size()-1; g++) {
      arrayCopy(genomes.get(layer-1).get(g), network.get(layer).weights.get(g));
    }
    network.get(layer).biases = genomes.get(layer-1).get(genomes.get(layer-1).size()-1);
  }
}

void graph(float pos, int total) {
  fitness = append(fitness, pos/total);
  stroke(255);
  for (int i=1; i<fitness.length; i++) {
    line(i, -fitness[i-1]+height, i+1, -fitness[i]+height);
  }
  textAlign(LEFT, BOTTOM);
  text(fitness[fitness.length-1], fitness.length+4, -fitness[fitness.length-1]+height);
  text(total, fitness.length+4, -fitness[fitness.length-1]+height+20);
  textAlign(CENTER);
}

void keyPressed() {
  if (key==' ') {
    newSim = true;
  }
}
