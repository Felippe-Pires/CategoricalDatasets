/*
* To change this license header, choose License Headers in Project Properties.
* To change this template file, choose Tools | Templates
* and open the template in the editor.
*/
package edu.uts.aai.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class is the main class of the DSFS algorithm, which consists of three components,
 * including intra-feature value outlierness computing (i.e., the delta function in our ICDM2016 paper),
 * the adjacent matrix of the feature graph, and dense subgraph discovery of the feature graph.
 * Note that we skip the value graph construction and go directly to construct the feature graph,
 * which can speed up the algorithm a bit.
 * @author Guansong Pang
 */
public class DSVL {
    
    private ArrayList<ValueCentroid> cpList;
    
    private ArrayList<ArrayList<Double>> vMatrix = new ArrayList<ArrayList<Double>>();
    
    private ArrayList<Integer> valFeatIDs = new ArrayList<Integer>();  //to store feature id to each feature value
    private ArrayList<String> valIdentifier = new ArrayList<String>();
    private ArrayList<Double> valIndWgts = new ArrayList<Double>();   //to store total weights for each feature value
    
    
    private int dim;
    private double[] attrWeight; //to store the weights of all the features
    /**
     *
     * @param cpList the list of coupled centroids: each centroid contains the co-occurrence frequency of each value with other values
     */
    public DSVL(ArrayList<ValueCentroid> cpList) {
        this.cpList = cpList;
        this.dim = cpList.size();
    }
    
    /**
     * the main method for calling Charikar greedy, sequential backward and Las Vegas based
     * dense subgraph discovery
     * @return the IDs of features to be removed
     */
    public String denseSubgraphDiscovery(int dataSize) {
        calcIntraFeatureValueOutlierness(dataSize);
//        adjacentMatrix(cpList);
        valueAdjacencyMatrix();
        ArrayList<String> discardVals = new ArrayList<String>();
        double[] den = charikarGreedySearchforValueGraph(discardVals);
        double max = Double.MIN_VALUE;
        int maxID = -1;
        for(int i = 0; i < discardVals.size(); i++) {
            if(den[i] > max) {
                max = den[i];
                maxID = i;
            }
        }
        System.out.println("MAX:"+(new DecimalFormat("#0.000000")).format(max)+" ");
        Plot.plotYPoints(den, 3, DSVL4ODUtils.dataSetName, DSVL4ODUtils.dataSetName, "Iteration", "Avg. Incoming Edge Weight");
        return discardVals.get(maxID);
    }
    
    public void valueOutliernessLearning(int dataSize) {
        calcIntraFeatureValueOutlierness(dataSize);
        valueAdjacencyMatrix();
        ArrayList<String> discardVals = new ArrayList<String>();
        charikarGreedySearchforValueGraph(discardVals);
        weightedDegree2Score();
    }
    
    /**
     * to calculate the outlierness of each feature value based on the extent the value frequent deviating from the mode frequency
     * @param dataSize the number of instances in the data set
     */
    public void calcIntraFeatureValueOutlierness(int dataSize) {
        double [] mFreq = new double[dim];
        for(int i = 0; i < cpList.size(); i++) {
            ValueCentroid cp = cpList.get(i);
            int len = cp.getCenList().size();
            double maxFreq = 0;
            for(int j = 0; j < len; j++) {
                ValueNode cen = cp.getCenList().get(j);
                double globalFreq = cen.globalFreq(i, j);
                if(globalFreq > maxFreq)
                    maxFreq = globalFreq;
            }
            mFreq[i] = maxFreq;
        }
        
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for(int i = 0; i < cpList.size(); i++) {
            ValueCentroid cp = cpList.get(i);
            int len = cp.getCenList().size();
            int count = 0;
            for(int j = 0; j < len; j++) {
                ValueNode cen = cp.getCenList().get(j);
                double globalFreq = cen.globalFreq(i, j);
                if(globalFreq == 0) {
                    continue;
                }
                double intra;
                intra = (Math.abs(globalFreq-mFreq[i])/mFreq[i] + (1-mFreq[i]/dataSize))/2.0; //mode absolute difference based. '1.0/dataSize' is used to avoid zero outlierness
                cen.setIntraOD(intra);
                count++;
            }
        }
    }
    /**
     * to generate the adjacent matrix for the value-value graph
     */
    public void valueAdjacencyMatrix() {
//        int nonzeroVal = 0;
        System.out.println();
        for(int i = 0; i < dim; i++) {
            ValueCentroid cp = cpList.get(i);
            int count = 0;
            for(int j = 0; j < cp.getCenList().size(); j++) {
                ValueNode cen = cp.getCenList().get(j);
                if(cen.globalFreq(i, j) == 0) { //skip zero-apperance values
                    
                    System.out.print("0.0"+" ");
                    continue;
                }
                ArrayList<Double> col = new ArrayList<Double>();
                double tmp=0;
                for(int k = 0; k < dim; k++) {
                    FeatureInfo ai = cen.getAttrList().get(k); //contain local co-occurrence informaiton, i.e., co-occurrnce frequency of the i-th and k-th feature values
                    FeatureInfo gai = cp.getGlobalCentroid().getAttrList().get(k); //contain global co-occurrencd information
                    int len = ai.NumofValue();
                    for(int l = 0; l < len; l++) {
                        if (k == cen.getOrgFeat() && gai.value(l) != 0) { //skip zero-appearance values
//                            if(j == l) {
//                                col.add(cen.getIntraOD());  // self-loop value graph
//                                tmp += cen.getIntraOD();
//                            }
//                            else
                            col.add(0.0);
                            System.out.print("0.0"+" ");
                            continue;
                        }/**/
                        double freq = ai.value(l);
                        double gFreq = gai.value(l);
                        double cenFreq = cen.globalFreq(i, j);
                        if(cenFreq != 0 && gFreq != 0) { //skip zero-appearance values
//                            double w = cen.getIntraOD() * cpList.get(k).getCenList().get(l).getIntraOD() * Math.log10(freq*1.0/(cenFreq * gFreq))/Math.log10(2.0);
                            double w = cen.getIntraOD() * cpList.get(k).getCenList().get(l).getIntraOD() * (freq*1.0/(cenFreq * gFreq));
//                            double w = cen.getIntraOD() * cpList.get(k).getCenList().get(l).getIntraOD(); //DSVLia
//                            double w =  (freq*1.0/(cenFreq * gFreq));  //DSVLie
                            
                            col.add(w);
                            tmp += w;
                            System.out.print(w+" ");
                        } else {
                            System.out.print("0.0"+" ");
                        }
                    }
                }
                vMatrix.add(col);
                //  valIndWgts.add(tmp);
                valFeatIDs.add(i);
                valIndWgts.add(tmp);
                count++;
                valIdentifier.add(i+"_"+j);
                cen.setWeightedDegree(tmp);
                System.out.println();
            }
        }
    }
    
    public void updateWeightedDegree() {
        for(int i = 0; i < dim; i++) {
            ValueCentroid cp = cpList.get(i);
            for(int j = 0; j < cp.getCenList().size(); j++) {
                ValueNode cen = cp.getCenList().get(j);
                double tmp = 0;
                if(cen.globalFreq(i, j) == 0) //skip zero-apperance values
                    continue;
                for(int k = 0; k < dim; k++) {
                    FeatureInfo ai = cen.getAttrList().get(k); //contain local co-occurrence informaiton, i.e., co-occurrnce frequency of the i-th and k-th feature values
                    FeatureInfo gai = cp.getGlobalCentroid().getAttrList().get(k); //contain global co-occurrencd information
                    int len = ai.NumofValue();
                    for(int l = 0; l < len; l++) {
                        if (k == cen.getOrgFeat() && gai.value(l) != 0) { //skip zero-appearance values
                            continue;
                        }
                        double freq = ai.value(l);
                        double gFreq = gai.value(l);
                        double cenFreq = cen.globalFreq(i, j);
                        if(cenFreq != 0 && gFreq != 0) { //skip zero-appearance values
//                            double w = cen.getIntraOD() * cpList.get(k).getCenList().get(l).getIntraOD() * Math.log10(freq*1.0/(cenFreq * gFreq))/Math.log10(2.0);
//                            double w = cen.getWeight() *  cpList.get(k).getCenList().get(l).getWeight(); // DSVLia
                            double w = cen.getWeight() *  cpList.get(k).getCenList().get(l).getWeight() //DSVLie and DSVL
                                    * (freq*1.0/(cenFreq * gFreq));
                            tmp += w;
                        }
                    }
                }
                cen.setWeightedDegree(tmp);
            }
        }
    }
    
    public void weightedDegree2Score() {
        double vol = 0;
        weightNormalization();
        updateWeightedDegree();
        for(int i = 0; i < dim; i++) {
            ValueCentroid cp = cpList.get(i);
            for(int j = 0; j < cp.getCenList().size(); j++) {
                ValueNode cen = cp.getCenList().get(j);
//                vol += cen.getWeightedDegree() * cen.getWeight();
                vol += cen.getWeightedDegree() ;
            }
        }
        
        attrWeight = new double[dim];
        for(int m=0; m < dim; m++) {
            attrWeight[m] = 1;
        }
        
        for(int i = 0; i < dim; i++) {
            ValueCentroid cp = cpList.get(i);
            for(int j = 0; j < cp.getCenList().size(); j++) {
                ValueNode cen = cp.getCenList().get(j);
//                double prob = cen.getWeightedDegree() * cen.getWeight() / vol;
//                double prob = cen.getWeightedDegree() / vol;
                double prob = cen.getWeight();
                cen.setOutlierScore(prob);
                System.out.print(prob+",");
                attrWeight[i] *= (1-prob); //compute attribute weight
            }
        }
        System.out.println();
        for(int m=0; m < dim; m++) {
            attrWeight[m] = 1-attrWeight[m];
        }
    }
    
    public void weightNormalization() {
        double totalWgt = 0;
        
        for(int i = 0; i < dim; i++) {
            ValueCentroid cp = cpList.get(i);
            for(int j = 0; j < cp.getCenList().size(); j++) {
                ValueNode cen = cp.getCenList().get(j);
                totalWgt += cen.getWeight();
            }
        }
        
        for(int i = 0; i < dim; i++) {
            ValueCentroid cp = cpList.get(i);
            for(int j = 0; j < cp.getCenList().size(); j++) {
                ValueNode cen = cp.getCenList().get(j);
                cen.setWeight(cen.getWeight() / totalWgt);
            }
        }
    }
    
    public Hashtable<Integer,Double> scoringTestInstances(Instances data) {        
        Hashtable<Integer,Double> scores = new Hashtable<Integer,Double>();        
        for(int j = 0; j < data.numInstances(); j++) {
            Instance inst = data.instance(j);
            double score=1;
            for(int i = 0; i <dim; i++) {
                ValueCentroid cp = cpList.get(i);
                int index = ((Double)inst.value(i)).intValue();
                ValueNode cen = cp.getCenList().get(index);
                double s = cen.getOutlierScore();
//                double s = cen.getOutlierDegree()* attrWeight[i]; // weighted sum
                //if(score < s)
                //   score = s;
//                score *= Math.pow(1-s,1);
                score *= Math.pow(1-s,attrWeight[i]);
//                score += s;
                
            }            
            score = 1-score;
            System.out.println(score);
            scores.put(j,score);
        }
        
        return scores;
    }
    /**
     * to search for the densest subgraph in indirected graphs by using Charikar's greedy method presented in the paper below
     * @incollection{charikar2000greedy,
     * title={Greedy approximation algorithms for finding dense components in a graph},
     *   author={Charikar, Moses},
     *   booktitle={Approximation Algorithms for Combinatorial Optimization},
     *   pages={84--95},
     *   year={2000},
     *   publisher={Springer}
     * }
     * @param discardFeats the list to store non-relevant feature ids
     * @return the density array that records all densities of all the subgraphs
     */
    public double[] charikarGreedySearchforValueGraph(ArrayList<String> discardVals) {
        int len = valIndWgts.size();
        int count = len;
        int id = 0;
        double[] den = new double[count];
        double accumulateDen = 0;
        StringBuilder sb = new StringBuilder();
//        System.out.print("Subgraph densities:");
        while(count > 0) {
            double density = 0;
            density = computeDensity(valIndWgts,count,sb);
            den[id++] = density;
            accumulateDen += density;
            discardVals.add(sb.toString());
//            System.out.println(sb.toString());
            double min = Double.MAX_VALUE;
            int mid = -1;
            for(int i = 0; i < valIndWgts.size(); i++) {
                double w = valIndWgts.get(i);
                if(w < min) {
                    min = w;
                    mid = i;
                }
            }
            removeOneFeatureValue(mid);
            String str = valIdentifier.remove(mid);
            sb.append(str + ",");
            String[] s = str.split("_");
            int featID = Integer.valueOf(s[0]);
            int valueID = Integer.valueOf(s[1]);
            ValueNode cen = cpList.get(featID).getCenList().get(valueID);
            cen.setWeight(accumulateDen);
            count--;
            
        }
//        System.out.println();
        return den;
    }
    
    
    /**
     * to remove one value from the value candidates
     * @param vid the id of the value to be removed
     */
    public void removeOneFeatureValue(int vid) {
        vMatrix.remove(vid);
        valIndWgts.remove(vid); //virtually remove the value
//            double fWgt = featIndWgts.get(fid);
        for(int k = 0; k < vMatrix.size(); k++) {
            ArrayList<Double> col = vMatrix.get(k);
            double vWgt = valIndWgts.get(k);
            double w = col.remove(vid);
//                double w = col.get(fid);
            valIndWgts.set(k, vWgt-w);
        }
    }
    
    /**
     * to compute the subgraph density using feature level array-list <code>featIndWgts</code>, i.e., average weight per node
     * @param edgeWeights the total incoming edge weights of individual feature values
     * @param valNum the number of values left
     * @param sb to store the non-relevant value ids
     * @return the subgraph density
     */
    public double computeDensity(ArrayList<Double> edgeWeights, int valNum, StringBuilder sb) {
        double density = 0;
        int len = edgeWeights.size();
        for(int i = 0; i < len; i++ ) {
            double w = edgeWeights.get(i);
            density += w;
        }
        density = density / (2*valNum);
//        System.out.print((new DecimalFormat("#0.000000")).format(density)+",");
        return density;
    }
    
    /**
     * to iteratively select top k relevant features
     * @param data a given data set
     * @param featNum the number of feature to be retained
     * @param path the directory for saving the newly generated data sets
     * @param name name of the given data set
     */
    public void featureSelection(Instances data, int featNum, String path, String name){
        Integer[] attrIDArray = new Integer[dim];
        
        for(int i = 0; i < dim; i++) {
            attrIDArray[i] = i;
        }
//        System.out.println();
        sortAndKeepIndex(attrIDArray);
        int len = attrIDArray.length;
        for(int i = 0; i < len; i++) {
            System.out.println(attrIDArray[i]+":"+attrWeight[attrIDArray[i]]+",");
        }
        System.out.println();
        
        int count = len;
        BufferedWriter writer = null;
        StringBuilder sb = new StringBuilder();
        Instances newData;
        int i=0;
        while(count > featNum) { //for a given fixed number
            int id = attrIDArray[i++]+1; //The Remove class assume index starts with 1 rather than 0.
//            System.out.print(id+","+count);
            sb.append(id).append(",");
            count--;
        }
        
        try {
            Remove remove = new Remove();
            System.out.println(sb.toString());
            remove.setAttributeIndices(sb.toString());
            remove.setInvertSelection(false);
            remove.setInputFormat(data);
            newData = Filter.useFilter(data, remove);
            writer = new BufferedWriter(new FileWriter(path+"\\"+name+"_"+String.format("%02d", count)+".arff"));
            writer.write(newData.toString());
            writer.flush();
            writer.close();
            sb.append(",");
        } catch (IOException ex) {
            Logger.getLogger(DSVL.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(DSVL.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                writer.close();
            } catch (IOException ex) {
                Logger.getLogger(DSVL.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    
    /**
     * to sort the features according to their weights while at the same time keeping track of their orginal indices
     * @param attrIDArray the original feature id array
     */
    public void sortAndKeepIndex(Integer[] attrIDArray) {
        Arrays.sort(attrIDArray, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Double.compare(attrWeight[o1], attrWeight[o2]);
            }
        });
    }
}
