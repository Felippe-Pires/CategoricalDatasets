
package edu.uts.aai.utils;

import java.util.ArrayList;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This class captures the interactions or couplings between feature values.
 * 
 * @author Guansong Pang
 */
public class ValueCentroid {
    
    private int orgFeat; // index of the dedicated feature
    private ArrayList<ValueNode> cenList = new ArrayList<>();   // to store centroids for a list of values in a feature, one centroid per value
    private static ValueNode globalCentroid; //to store frequency distribution of all the feature values
    
    /**
     * to allocate space for the centroids
     * @param data a given data set
     * @return list of centroids with empty contents
     */
    public ArrayList<ValueCentroid> initialCentroidList (Instances data) {
        ArrayList<ValueCentroid> cpList = new ArrayList<>();
        int d = data.numAttributes()-1;
        for (int i = 0; i < d; i++) {
            ValueCentroid cp = new ValueCentroid();
            cp.orgFeat = i;
            int card = data.attribute(i).numValues();
            //System.out.print(card+",");
            for(int k = 0; k < card; k++ ) {
                ValueNode cen = new ValueNode(i,k,data);
                String str = data.attribute(i).value(k);
                cen.setCategoricalContent(str);
                cp.cenList.add(cen);
            }
            cpList.add(cp);
        }
        return cpList;
    }
    
    /**
     * to generate a centroid for each feature value
     * @param cpList a empty list to save all the centroids
     * @param data a given data set
     * @return list of filled centroids
     */
    public ArrayList<ValueCentroid> generateCoupledCentroids(ArrayList<ValueCentroid> cpList, Instances data) {
        int size = data.numInstances();
        for(int i = 0; i < cpList.size(); i++) {
            ValueCentroid cp = cpList.get(i);
            for(int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                int index = new Double(inst.value(i)).intValue();
                ValueNode cen = cp.cenList.get(index);
                //System.out.println(j);
                cen.updateCentroid(inst);
            }
        }
        return cpList;
    }
    
    
    /**
     * to print out the centroid information of each value a given feature
     * @param cpList list of all the centroids
     * @param attrID a given feature id
     */
    public void printCoupledPatterns(ArrayList<ValueCentroid> cpList,int attrID) {
        ValueCentroid cp = cpList.get(attrID);
        for(int i = 0; i < cp.cenList.size(); i++) {
            ValueNode cen = cp.cenList.get(i);
            System.out.print("No."+i+":");
            cen.printCentroidInfo();
            System.out.println();
        }
    }
    
    
    /**
     * to obtain frequency distribution of all the feature values
     * @param cpList list of all the centroids
     * @param data a given data set
     */
    public void obtainGlobalCentroid(ArrayList<ValueCentroid> cpList, Instances data) {
        globalCentroid = new ValueNode(data);// only one global centroid is needed
        ValueCentroid cp = cpList.get(0);
        int len = cp.cenList.size();
        for(int j = 0; j < len; j++) {
            ValueNode cen = cp.cenList.get(j);
            globalCentroid.generateGlobalCentroid(cen);
        }
        // globalCentroid.printCentroidInfo();
    }
    
    /**
     *
     * @return the list of coupled centroids
     */
    public ArrayList<ValueNode> getCenList(){
        return this.cenList;
    }
    
    /**
     *
     * @return the frequencies of all feature values
     */
    public ValueNode getGlobalCentroid() {
        return globalCentroid;
    }
    
}
