/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if (is_initialized) {return;}
  
  num_particles = 10;

  std::default_random_engine generator;
  std::normal_distribution<double> distributionx (0.0,std[0]);
  std::normal_distribution<double> distributiony (0.0,std[1]);
  std::normal_distribution<double> distributiontheta (0.0,std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle = Particle();

    particle.id = i;
    particle.x = x + distributionx(generator);
    particle.y = y + distributiony(generator);
    particle.theta = theta + distributiontheta(generator);
    particle.weight = 1;
    particles.push_back(particle);
    weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine generator;
  std::normal_distribution<double> distributionx (0.0, std_pos[0]);
  std::normal_distribution<double> distributiony (0.0, std_pos[1]);
  std::normal_distribution<double> distributiontheta (0.0, std_pos[2]);

  for (int i = 0; i < particles.size(); ++i) {
    if (yaw_rate != 0) {
      particles[i].x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta = particles[i].theta + yaw_rate*delta_t;
    } else {
      particles[i].x = particles[i].x + velocity*cos(particles[i].theta);
      particles[i].y = particles[i].y + velocity*sin(particles[i].theta);
      // particle.theta = particle.theta
    }

    // add Gaussian noise
    particles[i].x += distributionx(generator);
    particles[i].y += distributiony(generator);
    particles[i].theta += distributiontheta(generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted, std::vector<Map::single_landmark_s> landmark_list) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.


  for (int i=0; i < predicted.size(); ++i) {
    LandmarkObs predictedLand = predicted[i];
    
    double closest_dist;
    int closest_landmark_id;

    for (int j=0; j < landmark_list.size(); ++j) {
      Map::single_landmark_s landmark = landmark_list[j];

      if (j==0 || dist(predictedLand.x, predictedLand.y, landmark.x_f, landmark.y_f) < closest_dist) {
        closest_dist = dist(predictedLand.x, predictedLand.y, landmark.x_f, landmark.y_f);
        predicted[i].id = landmark.id_i;
      }
    }
  }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double weights_sum = 0.0;

  for (int i=0; i<particles.size(); ++i){
    std::vector<LandmarkObs> obs_trans;

    for (int j=0; j<observations.size(); ++j){
      // transform observation to maps coordinates
      LandmarkObs part_obs;
      part_obs.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      part_obs.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
      obs_trans.push_back(part_obs);
    }

    dataAssociation(obs_trans, map_landmarks.landmark_list);

    double prob = 1.0;
    for (int j=0; j<obs_trans.size(); ++j){
      for (int k=0; k < map_landmarks.landmark_list.size(); ++k) {
        if (map_landmarks.landmark_list[k].id_i == obs_trans[j].id) {
          Map::single_landmark_s ldmk = map_landmarks.landmark_list[k];
          
          // exp {-1/2[(X-x)^2/sig_x^2 + (Y-y)^2/sig_y_2]}
          // -----------------------------------------------
          //       sqrt(2*M_PI*(sig_x_2 * sig_y_2))

          prob *= exp(-(1/2.) * ( pow(obs_trans[j].x - ldmk.x_f, 2) / pow(std_landmark[0], 2) + pow(obs_trans[j].y - ldmk.y_f, 2) / pow(std_landmark[1], 2) ));
          prob = prob/sqrt(2. * M_PI * (pow(std_landmark[0], 2) * pow(std_landmark[1], 2))); 
          if (prob == 0.0) {
            std::cout << "obs_trans[j].x" << obs_trans[j].x << std::endl;
            std::cout << "ldmk.x_f" << ldmk.x_f << std::endl;
            std::cout << "std_landmark[0]" << std_landmark[0] << std::endl;
            std::cout << "obs_trans[j].y" << obs_trans[j].y << std::endl;
            std::cout << "ldmk.y_f" << ldmk.y_f << std::endl;
            std::cout << "std_landmark[1]" << std_landmark[1] << std::endl;
          }
        }
      }
    }

    weights[i] = prob;
    particles[i].weight = prob;
    weights_sum += prob;
  }

  if (weights_sum == 0.0) {
    exit(0);
  }

  for (int i = 0; i < weights.size(); ++i) {
    weights[i] = weights[i]/weights_sum;
    particles[i].weight = weights[i];
  }
}

void ParticleFilter::resample() {
  int N = weights.size();
  // std::cout << "resample" << std::endl;
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  double w_max = -1.0;
  for (int i = 0; i < N; ++i) {
    if (weights[i] > w_max) {
      w_max = weights[i];
    }
  }

  
  double beta = 0.0;
  std::vector<Particle> temp_particles;

  int index = rand()%N;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> u_distrib(0.0,1.0);

  for (int i = 0; i < N; ++i) {
    beta = beta + 2 * w_max * u_distrib(generator);
    
    while (weights[index] < beta){
      beta = beta - weights[index];
      index = index + 1;
      index = index%N;
    }
    temp_particles.push_back(particles[index]);
  }

  particles = temp_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
