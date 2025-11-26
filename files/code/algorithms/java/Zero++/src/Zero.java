
/*
*    Zero.java
*    Copyright (C) 2012 University of Waikato, Hamilton, New Zealand
*
*    Revised by Guansong Pang on 14/02/2014
*
*/
import weka.classifiers.RandomizableClassifier;

import java.util.Enumeration;
import java.util.Random;
import java.util.ArrayList;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import weka.core.Attribute;
import weka.core.FastVector;


/**
 * <!-- globalinfo-start -->
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Guansong Pang revised it fro isolationforest in weka
 * @version $Revision: 9532 $
 */
public class Zero extends RandomizableClassifier implements TechnicalInformationHandler, Serializable {
    
    // For serialization
    private static final long serialVersionUID = 5586674623147772788L;
    
    // The set of trees
    protected Tree[] m_trees = null;
    
    // The number of trees
    protected int m_numTrees = 50;
    
    // The subsample size
    protected int m_subsampleSize = 8;
    
    protected int self_seed = 1;
    
    protected int dim_subspace = 2;
    /**
     * Returns a string describing this filter
     */
    public String globalInfo() {
        
        return "Implements the zero++ method for anomaly detection in large-scale"
                + "categorical data as well as numeric data. "
                + "\n\nFor more information, see:\n\n"
                + getTechnicalInformation().toString();
    }
    
    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        
        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Guansong Pang and Kai Ming Ting and David Albrecht and Huidong Jin");
        result.setValue(Field.TITLE, "ZERO++: Harnessing the Power of Zero Appearances to DetectAnomalies in Large-scale Data Sets");
        result.setValue(Field.JOURNAL, "Journal of Artificial Intelligence Research (JAIR)");
        result.setValue(Field.YEAR, "2016");
        result.setValue(Field.PUBLISHER, "AAAI");
        
        return result;
    }
    
    /**
     * Returns the Capabilities of this filter.
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        //result.disableAll();
        
        // attributes
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);   // change this property to improve iForest's capabilities !!!!!
        
        // class
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.EMPTY_NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        // instances
        result.setMinimumNumberInstances(0);
        
        return result;
    }
    
    /**
     * Returns brief description of the classifier.
     */
    public String toString() {
        
        if (m_trees == null) {
            return "No model built yet.";
        } else {
            return "ZERO++ for anomaly detection (" +
                    m_numTrees + ", " + m_subsampleSize + ")";
        }
    }
    
    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String numTreesTipText() {
        
        return "The number of subsamples to use in the forest.";
    }
    
    /**
     * Get the value of numTrees.
     *
     * @return Value of numTrees.
     */
    public int getNumTrees() {
        
        return m_numTrees;
    }
    
    /**
     * Set the value of numTrees.
     *
     * @param k value to assign to numTrees.
     */
    public void setNumTrees(int k) {
        
        m_numTrees = k;
    }
    
    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String subsampleSizeTipText() {
        
        return "The size of the subsample used to build each model.";
    }
    
    /**
     * Get the value of subsampleSize.
     *
     * @return Value of subsampleSize.
     */
    public int getSubsampleSize() {
        
        return m_subsampleSize;
    }
    
    /**
     * Set the value of subsampleSize.
     *
     * @param n value to assign to subsampleSize.
     */
    public void setSubsampleSize(int n) {
        
        m_subsampleSize = n;
    }
    
    /**
     * Lists the command-line options for this classifier.
     *
     * @return an enumeration over all possible options
     */
    @SuppressWarnings("unchecked")
    public Enumeration listOptions() {
        
        Vector newVector = new Vector();
        
        newVector.addElement(new Option(
                "\tThe number of subsamples in the zero++ (default 50).", "I", 1,
                "-I <number of subsample>"));
        
        newVector.addElement(new Option(
                "\tThe subsample size (default 8).", "N", 1,
                "-N <the size of the subsample>"));
        
        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        
        return newVector.elements();
    }
    
    /**
     * Gets options from this classifier.
     *
     * @return the options for the current setup
     */
    @SuppressWarnings("unchecked")
    public String[] getOptions() {
        Vector result;
        String[] options;
        int i;
        
        result = new Vector();
        
        result.add("-I");
        result.add("" + getNumTrees());
        
        result.add("-N");
        result.add("" + getSubsampleSize());
        
        options = super.getOptions();
        for (i = 0; i < options.length; i++) {
            result.add(options[i]);
        }
        
        return (String[]) result.toArray(new String[result.size()]);
    }
    
    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start -->
     * <!-- options-end -->
     *
     * @param options
     *            the list of options as an array of strings
     * @throws Exception
     *             if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String tmpStr;
        
        tmpStr = Utils.getOption('I', options);
        if (tmpStr.length() != 0) {
            m_numTrees = Integer.parseInt(tmpStr);
        } else {
            m_numTrees = 50;
        }
        
        tmpStr = Utils.getOption('N', options);
        if (tmpStr.length() != 0) {
            m_subsampleSize = Integer.parseInt(tmpStr);
        } else {
            m_subsampleSize = 8;
        }
        
        super.setOptions(options);
        
        Utils.checkForRemainingOptions(options);
    }
    
    /**
     * Builds the forest.
     * @param data the training instances
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        // Can classifier handle the data?
        getCapabilities().testWithFail(data);
        
        // Reduce subsample size if data is too small
        if (data.numInstances() < m_subsampleSize) {
            m_subsampleSize = data.numInstances();
        }
        
        // Generate trees
        //  System.out.println(m_subsampleSize+":"+m_numTrees);
        m_trees = new Tree[m_numTrees];
        Random r = (data.numInstances() > 0) ?
                getRandomNumberGenerator(self_seed,data) : new Random(self_seed);
        
        int datatype = numericalOrNominalOrMixed(data);
        
        for (int i = 0; i < m_numTrees; i++) {
//             data.randomize(r);
//            ArrayList<Integer> al = new ArrayList();
//            for(int j = 0; j < data.numAttributes()-1; j++) {
//                al.add(j);
//            }
//            m_trees[i] = new Tree();
//            if(datatype == 1)
//                m_trees[i].buildNumericDataModelPermutation(new Instances(data, 0, m_subsampleSize), al,dim_subspace);
//            else if(datatype == 2)
//                m_trees[i].buildFrequencyTablePermutation(new Instances(data, 0, m_subsampleSize),al,dim_subspace);
            
            data.randomize(r);
            FastVector al = new FastVector();
            for(int j = 0; j < data.numAttributes()-1; j++) {
                // al.addElement(r.nextInt(data.numAttributes()-1));
                al.addElement(j);
            }
            for (int j = data.numAttributes()- 2; j > 0; j--)
                al.swap(j, r.nextInt(j + 1));
            
            m_trees[i] = new Tree();
            if(datatype == 1)
                m_trees[i].buildNumericDataModel(new Instances(data, 0, m_subsampleSize), al);
            else if(datatype == 2)
                m_trees[i].buildFrequencyTable(new Instances(data, 0, m_subsampleSize),al);
            else if(datatype == 3)
                m_trees[i].buildMixedDataModel(new Instances(data, 0, m_subsampleSize), al);/**/
        }
    }
    
    
    
    
    
    protected void setSubspaceDim(int num) {
        dim_subspace = num;
    }
    
    /**
     * to compute the outlier scores of test instances
     * @param data the test instance set
     * @return the outlier scores of the test instances
     */
    public Hashtable<Integer,Double> scoringTestInstances(Instances data) {
        Hashtable<Integer,Double> scores = new Hashtable();
        int dataType = numericalOrNominalOrMixed(data);
        for(int idx = 0; idx < data.numInstances(); idx++) {
            Instance inst = data.instance(idx);
            double zeros = 0;
            for (int i = 0; i < m_trees.length; i++) {
//             if(dataType == 1)
//                    linkStrength +=  m_trees[i].numericalDataLinkStrengthPermutation(inst);
//             else if(dataType == 2)
//                    linkStrength +=  m_trees[i].nominalDatalinkStrengthPermutation(inst);
                if(dataType == 1)
                    zeros +=  m_trees[i].numericalDataZero(inst);
                else if(dataType == 2)
                    zeros +=  m_trees[i].nominalDataZero(inst);
                else if(dataType == 3)
                    zeros += m_trees[i].mixedDataZero(inst);/**/
            }
            zeros /= (double) m_trees.length;
//             System.out.println(zeros);
            /* if(inst.value(inst.numAttributes()-1) == 0) {
            avgZerosPos += linkStrength;
            posCount ++;
            }
            else
            avgZerosNeg  += linkStrength;*/
            scores.put(idx, zeros);
            /* avgPathLength /= (double) m_trees.length;
            scores.put(idx, avgPathLength);*/
        }
        // System.out.println(avgZerosPos/posCount+","+avgZerosNeg/(data.numInstances()-posCount));
        return scores;
    }
    
    /**
     * to test the data type of an input data set
     * @param data the input data set
     * @return the data type, return 1 if it is numeric data, return 2 if it is nominal data, return 3 if it is mixed-attribute data
     */
    public static int numericalOrNominalOrMixed(Instances data) {
        int indicator1 = 0;
        int indicator2 = 0;
        for(int i = 0; i < data.numAttributes()-1; i++) {
            if(data.attribute(i).isNumeric())
                indicator1 = 1;
            else if(data.attribute(i).isNominal())
                indicator2 = 2;
        }
        return (indicator1 + indicator2);
    }
    
    public void setSelfGeneratedRandomSeed(int seed) {
        self_seed = seed;
    }
    
    /**
     * Main method for this class.
     */
    public static void main(String[] args) {
        
        runClassifier(new Zero(), args);
    }
    
    public Random getRandomNumberGenerator(long seed,Instances data) {
        
        Random r = new Random(seed);
        StringBuffer text = new StringBuffer();
        Instance inst = data.instance(r.nextInt(data.numInstances()));
        
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i > 0)
                text.append(",");
            // Instance isnt = data.instance(i);
            Attribute attr = inst.attribute(i);
            if (inst.isMissing(i)) {
                text.append("?");
            } else {
                switch (attr.type()) {
                    case Attribute.NOMINAL:
                        text.append(Utils.doubleToString(inst.value(i), 2));
                        break;
                    case Attribute.STRING:
                    case Attribute.DATE:
                    case Attribute.RELATIONAL:
                        text.append(Utils.quote(inst.stringValue(i)));
                        break;
                    case Attribute.NUMERIC:
                        text.append(Utils.doubleToString(inst.value(i), 6));
                        // text.append(i);
                        break;
                    default:
                        throw new IllegalStateException("Unknown attribute type");
                }
            }
        }
        long newSeed = text.toString().hashCode();
        r.setSeed(newSeed + seed);
        return r;
    }
    
    
    /**
     * Inner class for building and using an isolation tree.
     */
    protected class Tree implements Serializable {
        
        // For serialization
        private static final long serialVersionUID = 7786674623147772711L;
        
        // The size of the node
        protected int m_size;
        
        // The split attribute
        protected int m_a;
        
        // The split point
        protected double m_splitPoint;
        
        // The label distribution container
        protected ArrayList<HashMap<String,Integer>> labFreDistribution;
        
        protected HashMap<String,ArrayList<Integer>> invertedIndexMap;
        
        //the label-frequency hash map
        protected ArrayList<HashMap<String,Integer>> freqTable;
        
        //the label-numerical_values hash map
        protected HashMap<String,String> freqTableForNominalNumericCombined;
        
        protected String m_split_label;
        
        //the subsamples
        protected Instances samples;
        
        protected double m_uncertainty;
        
        protected ArrayList<List> powerSet;
        
        //the randomized attribute order
        protected FastVector attrOrder;
        
        //the standard deviations of all attributes
        protected double[] attrStds;
        
        protected double[] attrMeans;
        
        //the data type of 2-D subspace
        protected int[] subspaceAttrType;
        
        // The successors
        protected Tree[] m_successors;
        
        public Tree() {
            
        }
        
        /**
         * to construct an inverted index for all the labels of the subsamples
         * @param data the subsample set
         */
        protected void buildLabelInvertedIndex(Instances data) {
            m_size = data.numInstances();
            invertedIndexMap = new HashMap<String,ArrayList<Integer>>();
            int numAttr = data.numAttributes();
            for(int i = 0 ; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                for(int j = 0; j < numAttr-1; j++) {
                    double idx = inst.value(j);
                    String label = String.valueOf("A"+String.valueOf(j)+"_"+String.valueOf(idx));
                    ArrayList<Integer> postings = invertedIndexMap.get(label);
                    if(postings == null) {
                        postings = new ArrayList();
                        postings.add(i);
                    } else {
                        postings.add(i);
                    }
                    invertedIndexMap.put(label, postings);
                }
            }
        }
        
        /**
         * to construct an frequency table for all the label-pairs of the subsamples
         * @param data the subsample set
         * @param al the randomized attribute order
         */
        protected void buildFrequencyTable(Instances data,FastVector al) {
            // System.out.println(data.numInstances());
            m_size = data.numInstances();
            freqTable = new ArrayList<HashMap<String,Integer>>();
            for (int i = 0; i < al.size(); i++) {
                if(!data.attribute(i).isNominal() && !data.attribute(i).isString())
                    continue;
                HashMap<String,Integer> hm = new HashMap<String,Integer>();
                for(int j = 0; j < data.numInstances(); j++) {
                    Instance inst = data.instance(j);
                    //  String str = this.getLabelsofJointAttributes(inst, i,al,true,false);
                    String str = this.getLabelsofMultipleJointAttributes(inst, i,al,dim_subspace);
                    //System.out.println(str);
                    // Integer fre = freqTable.get(str);
                    // freqTable.put(str, fre == null? 1 : ++fre);
                    Integer fre = hm.get(str);
                    hm.put(str, fre == null? 1 : ++fre);
                }
                freqTable.add(hm);
                
                // System.out.println("==========="+i+"th attr===========");
            }
            //System.out.println(freqTable.size());
            attrOrder = al;
            
        }
        
        protected void buildFrequencyTablePermutation(Instances data,ArrayList al,int m) {
            // System.out.println(data.numInstances());
            // System.out.println("###");
            m_size = data.numInstances();
            powerSet = new ArrayList();
            freqTable = new ArrayList<HashMap<String,Integer>>();
            OrderedPowerSet ops = new OrderedPowerSet(al);
            //   for(int i = 1; i <= al.size(); i++) {
            // List set = ops.getPermutationsList(i);
            List set = ops.getPermutationsList(m);
            HashMap<String,Integer> hm = new HashMap<String,Integer>();
            for(int j = 0; j < set.size(); j++) {
                Object o = set.get(j);
                //System.out.println(o.toString());
                Object arr[] = ((Set)o).toArray();
                for(int k = 0; k < data.numInstances(); k++) {
                    Instance inst = data.instance(k);
                    String str = this.getLabelsofJointAttributes(inst, arr);
                    //  System.out.println(str);
                    Integer fre = hm.get(str);
                    hm.put(str, fre == null? 1 : ++fre);
                }
                // System.out.println("==========="+i+"th attr===========");
            }
            freqTable.add(hm);
            powerSet.add(set);
        }/**/
        
        
        /**
         * to calculate the standard deviations of all numerical attributes
         * @param data the subsample data
         * @param al the randomized attribute order
         */
        protected void buildNumericDataModel(Instances data,FastVector al) {
            samples = data;
            m_size = data.numInstances();
            attrOrder = al;
            attrStds = new double[data.numAttributes()-1];
            attrMeans = new double[data.numAttributes()-1];
            for(int i = 0; i < al.size(); i++) {
                int idx = (Integer)al.elementAt(i);
                attrStds[idx] = Math.sqrt(samples.variance(idx));
                attrMeans[idx] = samples.meanOrMode(idx);
                
            }
        }
        
        /**
         * to calculate the standard deviations of all numerical attributes
         * @param data the subsample data
         * @param al the randomized attribute order
         */
        protected void buildNumericDataModelPermutation(Instances data,ArrayList al,int m) {
            
            samples = data;
            m_size = data.numInstances();
            powerSet = new ArrayList();
            freqTable = new ArrayList<HashMap<String,Integer>>();
            OrderedPowerSet ops = new OrderedPowerSet(al);
            // for(int i = 1; i <= al.size(); i++) {
            //   List set = ops.getPermutationsList(i);
            List set = ops.getPermutationsList(m);
            powerSet.add(set);
            //System.out.println(set.size());
            //  }
            attrStds = new double[data.numAttributes()-1];
            attrMeans = new double[data.numAttributes()-1];
            for(int i = 0; i < al.size(); i++) {
                int idx = (Integer)al.get(i);
                attrStds[idx] = Math.sqrt(samples.variance(idx));
                attrMeans[idx] = samples.meanOrMode(idx);
            }
            //  System.out.println("##");
        }
        
        /**
         * to compute the frequencies or the standard deviations for nominal or numerical attributes
         * @param data the subsample
         * @param al the randomized attribute order
         */
        protected void buildMixedDataModel(Instances data,FastVector al) {
            samples = data;
            m_size = data.numInstances();
            attrOrder = al;
            attrStds = new double[samples.numAttributes()-1];
            attrMeans = new double[data.numAttributes()-1];
            // freqTable = new HashMap<String,Integer>();
            freqTable = new ArrayList<HashMap<String,Integer>>();
            freqTableForNominalNumericCombined = new HashMap<String,String>();
            // subspaceAttrType = new int[samples.numAttributes()-2];
            subspaceAttrType = new int[samples.numAttributes()-1];
            // for(int i = 0; i < al.size()-1; i++) {
            for(int i = 0; i < al.size(); i++) {
                int idx1 = (Integer)al.elementAt(i);
                if(samples.attribute(idx1).isNumeric()) {
                    attrStds[idx1] = Math.sqrt(samples.variance(idx1));
                    attrMeans[idx1] = samples.meanOrMode(idx1);
                    int idx2 = (Integer)al.elementAt( (i+1) % al.size());
                    if (samples.attribute(idx2).isNominal()) {
                        subspaceAttrType[i] = 2; // "2" indicates a numeric_nominal attribute combination
                        for(int j = 0; j < samples.numInstances(); j++) {
                            Instance inst = samples.instance(j);
                            double val =  inst.value(idx1);
                            //  String str = this.getLabelsofJointAttributes(inst, i+1,al,false,false);
                            String str = this.getLabelsofMultipleJointAttributes(inst, i+1,al,dim_subspace);
                            str = "A"+String.valueOf(idx1)+"_"+str; //this helps avoid the frequency counting conflicts of numeric-nominal-numeric consecutive attributes
                            String numValues = freqTableForNominalNumericCombined.get(str);
                            freqTableForNominalNumericCombined.put(str, numValues == null? String.valueOf(val) : numValues.concat(","+String.valueOf(val)));
                        }
                    }
                    else if(samples.attribute(idx2).isNumeric()) {
                        subspaceAttrType[i] = 1; // "1" indicates a numeric_numeric attribute combination
                        
                        // compute the std if idx2 is the last attribute, otherwise not necessarily to repeatedly compute
                        //the std for one attribute, as it will be computed at next step
                        // if( i  == al.size() - 2)
                        //    attrStds[idx2] = Math.sqrt(samples.variance(idx2));
                    }
                }
                if(samples.attribute(idx1).isNominal()) {
                    int idx2 = (Integer)al.elementAt( (i+1) % al.size());
                    if (samples.attribute(idx2).isNominal()) {
                        subspaceAttrType[i] = 3; // "3" indicates a nominal_nominal attribute combination
                        HashMap<String,Integer> hm = new HashMap<String,Integer>();
                        for(int j = 0; j < samples.numInstances(); j++) {
                            Instance inst = samples.instance(j);
                            //  String str = this.getLabelsofJointAttributes(inst, i,al,true,false);
                            String str = this.getLabelsofMultipleJointAttributes(inst, i,al,dim_subspace);
                            //  Integer fre = freqTable.get(str);
                            //   freqTable.put(str, fre == null? 1 : ++fre);
                            Integer fre = hm.get(str);
                            hm.put(str, fre == null? 1 : ++fre);
                        }
                        freqTable.add(hm);
                    }
                    else if(samples.attribute(idx2).isNumeric()) {
                        subspaceAttrType[i] = 4; // "2" indicates a nominal_numeric attribute combination
                        
                        // compute the std if idx2 is the last attribute, otherwise not necessarily to repeatedly compute
                        //the std for one attribute, as it will be computed at next step
                        // if(i == al.size() - 2)
                        //    attrStds[idx2] = Math.sqrt(samples.variance(idx2));
                        
                        for(int j = 0; j < samples.numInstances(); j++) {
                            Instance inst = samples.instance(j);
                            double val =  inst.value(idx2);
                            // String str = this.getLabelsofJointAttributes(inst, i,al,false,false);
                            String str = this.getLabelsofMultipleJointAttributes(inst, i,al,dim_subspace);
                            String numValues = freqTableForNominalNumericCombined.get(str);
                            freqTableForNominalNumericCombined.put(str, numValues == null? String.valueOf(val) : numValues.concat(","+String.valueOf(val)));
                        }
                    }
                }
            }
            //  System.out.println("##");
        }
        /**
         * to generate labels of joint attributes. to generate labels of single attributes by default
         * @param inst the input instance
         * @param beginAttrIndex the index of the first attribute in the joint attributes
         * @param al the randomized order of attributes
         * @param pairJoint to determine whether we want to obtain labels of pair joint attributes
         * @param tripleJoint to determine whether we want to obtain labels of triple joint attributes
         * @return labels of joint attributes
         */
        protected String getLabelsofJointAttributes(Instance inst, int beginAttrIndex, FastVector al,boolean pairJoint, boolean tripleJoint) {
            int idx1 = (Integer) al.elementAt(beginAttrIndex);
            int index1 = ((Double)inst.value(idx1)).intValue(); // return the index of the label of this instance in a specific categorical attribute
            // String label = "A"+idx1+"_"+index1;
            String label = beginAttrIndex+"_"+"A"+idx1+"_"+index1;
            // String str = inst.attribute(idx1).value(index1); // Get the label in a set of values in a specific categorical attribute
            if(pairJoint == true) {
                int idx2 = ((Integer) al.elementAt(++beginAttrIndex))%(inst.numAttributes()-1);
                int index2 = ((Double)inst.value(idx2)).intValue();
                //String str1 = inst.attribute(idx1).value(index1); // Get the label in a set of values in a specific categorical attribute
                //String str2 = inst.attribute(idx2).value(index2);
                if(tripleJoint == true) {
                    int idx3 = ((Integer) al.elementAt(++beginAttrIndex)) % (inst.numAttributes()-1);
                    int index3 = ((Double)inst.value(idx3)).intValue();
                    //  String str3 = inst.attribute(idx3).value(index3);
                    // str = "A"+idx1+"_"+str1.trim()+"A"+idx2+"_"+str2.trim()+"A"+idx3+"_"+str3.trim();
                    //label = "A"+idx1+"_"+index1+"A"+idx2+"_"+index2+"A"+idx3+"_"+index3;
                    label = beginAttrIndex+"_"+"A"+idx1+"_"+index1+"A"+idx2+"_"+index2+"A"+idx3+"_"+index3;
                } else {
                    // label = "A"+idx1+"_"+index1+"A"+idx2+"_"+index2;
                    label = beginAttrIndex+"_"+"A"+idx1+"_"+index1+"A"+idx2+"_"+index2;
                }
            }
            //System.out.println(label);
            return label;
        }
        
        protected String getLabelsofJointAttributes(Instance inst, Object arr[]) {
            String label = "";
            for (int i = 0; i < arr.length; i++) {
                int idx = (Integer)arr[i];
                int index = ((Double)inst.value(idx)).intValue();
                label = label.concat("A"+idx+"_"+index+"_");
            }
            //System.out.println(label);
            return label;
        }
        
        protected String getLabelsofMultipleJointAttributes(Instance inst, int beginAttrIndex, FastVector al, int numJAtts) {
            String label = "";
            for(int i = beginAttrIndex; i < al.size(); i++) {
                if(numJAtts == 0)
                    break;
                int idx1 = (Integer) al.elementAt(i % al.size());
                int index1 = ((Double)inst.value(idx1)).intValue();
                label = label.concat(i+"_"+"A"+idx1+"_"+index1+"_");
                numJAtts--;
            }
            
            //System.out.println(label);
            return label;
        }
       
        
        /**
         * to calculate the zero numbers of an instance for nominal data
         * @param inst an input test instance
         * @return the zero numbers of the test instance
         */
        protected double nominalDataZero(Instance inst) {
            double score;
            double zeros = 0;
            for(int j = 0; j <attrOrder.size(); j++ ) {
                //  String str = this.getLabelsofJointAttributes(inst, j, attrOrder,true,false);
                String str = this.getLabelsofMultipleJointAttributes(inst, j, attrOrder,dim_subspace);
                // Integer intersectCounts = freqTable.get(str);
                HashMap<String,Integer> hm = freqTable.get(j);
                Integer intersectCounts = hm.get(str);              
                if(intersectCounts == null) {
                    intersectCounts = 0;
                    zeros++;
                }
               
            }
            score = 1 - zeros / attrOrder.size();
            return score;
        }
        
        
        
        /**
         * to calculate the zero numbers of an instance for nominal data
         * @param inst an input test instance
         * @return the zero numbers of the test instance
         */
        protected double nominalDataZeroPermutation(Instance inst) {
            double zeros = 0;
            double p=0;
            for(int i = 0; i < powerSet.size(); i++) {
                List set = powerSet.get(i);
                HashMap<String,Integer> hm = freqTable.get(i);;
                Random r = new Random(6);
                for(int j=0; j < 2; j++) {
                    Object o = set.get(r.nextInt(set.size()));
                    Object arr[] = ((Set)o).toArray();
                    String str = this.getLabelsofJointAttributes(inst, arr);
                    Integer counts = hm.get(str);
                    
                    if(counts == null) {
                        zeros++;
                        
                    }
                }
            }
            return zeros;
        }
        /**
         * to calculate the zero numbers of an instance for numeric data
         * @param inst an input test instance
         * @return the zero numbers of the test instance
         */
        protected double numericalDataZero(Instance inst) {
            double score = 1;
            int j ;
            double zeros = 0;
            // for(j = 0; j <attrOrder.size()-1; j++ ) {
            for(j = 0; j <attrOrder.size(); j++ ) {
                //double linkStrength = this.numericalDataSubspaceLinkStrength(inst, j);
                double linkStrength = this.numericalDataMultipleSubspaceZero(inst, j, dim_subspace);
                if(linkStrength == 0)  {
                    zeros++;
                }
                
            }
            
            score = 1 - zeros / attrOrder.size();
            
            return score;
        }
        
        
        protected double numericalDataZeroPermutation(Instance inst) {
            double score = 1;
            double zeros = 0;
            for(int i = 0; i < powerSet.size(); i++) {
                List set = powerSet.get(i);
                for(int j = 0; j < set.size(); j++) {
                    Object o = set.get(j);
                    Object arr[] = ((Set)o).toArray();
                    boolean flag = false;
                    for (Object arr1 : arr) {
                        int trueAttID = (Integer) arr1;
                        double attrValue =  inst.value(trueAttID);
                        if(Math.abs(attrMeans[trueAttID]-attrValue) > 3*attrStds[trueAttID]) {
                            flag = true;
                            break;
                        }
                    }
                    if(flag)
                        zeros++;
                }
            }
            score = 1 - zeros / powerSet.size();
            return score;
        }
        /**
         * to calculate the zero numbers of an instance for mixed-attribute data
         * @param inst an input test instance
         * @return the zero numbers of the test instance
         */
        protected double mixedDataZero(Instance inst) {
            double score = 0;
            double zeros = 0;
            // System.out.println("####");
            //for(int j = 0; j <attrOrder.size()-1; j++ ) {
            for(int j = 0; j <attrOrder.size(); j++ )  {
                if(numericalDataSubspaceZero(inst,j) == 0)
                    zeros++;
                else if(subspaceAttrType[j] == 2 || subspaceAttrType[j] == 4)
                    if(mixedDataSubspaceZero(inst,j) == 0)
                        zeros++;
                    else if(subspaceAttrType[j] == 3) {
                        //String str = this.getLabelsofJointAttributes(inst, j, attrOrder,true,false);
                        String str = this.getLabelsofMultipleJointAttributes(inst, j, attrOrder,dim_subspace);
                        HashMap<String,Integer> hm = freqTable.get(j);
                        Integer linkStrength = hm.get(str);
                        //Integer linkStrength = freqTable.get(str);
                        if(linkStrength == null)
                            zeros++;
                    }
            }
            score = 1 - zeros / attrOrder.size();
            // System.out.println(score);
            return score;
        }
        
        /**
         * to calculate the 2-D zero numbers of an instance from numeric data sets
         * @param inst a test instance
         * @param beginAttrIndex the index of the first attribute of the 2-D subspace
         * @return  the 2-D zero numbers of the instance
         */
        protected double numericalDataSubspaceZero(Instance inst, int beginAttrIndex) {
            double zero = 0;
            int idx1 = (Integer) attrOrder.elementAt(beginAttrIndex);
            double attr1 =  inst.value(idx1);
            int idx2 = (Integer) attrOrder.elementAt(++beginAttrIndex % attrOrder.size());
            double attr2 = inst.value(idx2);
            if(Math.abs(attrMeans[idx1]-attr1) <= 3*attrStds[idx1] && Math.abs(attrMeans[idx2]-attr2) <= 3*attrStds[idx2])
                zero = samples.numInstances();
            return zero;
            
        }
        
        /**
         * to calculate the 2-D zero numbers of an instance from numeric data sets
         * @param inst a test instance
         * @param beginAttrIndex the index of the first attribute of the 2-D subspace
         * @return  the 2-D zero numbers of the instance
         */
        protected double numericalDataMultipleSubspaceZero(Instance inst, int beginAttrIndex, int numJAtts) {
            // System.out.println("#");
            double linkStrength = samples.numInstances();
            boolean flag = false;
            while(numJAtts > 0) {
                int id = beginAttrIndex % attrOrder.size();
                int trueAttID = (Integer) attrOrder.elementAt(id);
                double attrValue =  inst.value(trueAttID);
                //if(Math.abs(attrMeans[trueAttID]-attrValue) > attrStds[trueAttID]) {
                if(Math.abs(attrMeans[trueAttID]-attrValue) > 3*attrStds[trueAttID]) {
                    flag = true;
                    break;
                }
                beginAttrIndex++;
                numJAtts--;
            }
            if(flag == true)
                linkStrength = 0;
            //  System.out.print(numJAtts+","+flag+","+linkStrength+",");
            return linkStrength;
        }
        
        /**
         * to calculate the 2-D zero numbers of an instance from mixed-attribute data sets
         * @param inst a test instance
         * @param beginAttrIndex the index of the first attribute of the 2-D subspace
         * @return  the 2-D zero numbers of the instance
         */
        protected double mixedDataSubspaceZero(Instance inst, int beginAttrIndex) {
            double strength = 0;
            if(subspaceAttrType[beginAttrIndex] == 2) { //for a numeric_nominal attribute combination
                // String str = this.getLabelsofJointAttributes(inst, beginAttrIndex+1,attrOrder,false,false);
                String str = this.getLabelsofMultipleJointAttributes(inst, beginAttrIndex+1,attrOrder,dim_subspace);
                int idx1 = (Integer)attrOrder.elementAt(beginAttrIndex % attrOrder.size());
                str = "A"+String.valueOf(idx1)+"_"+str;
                if(freqTableForNominalNumericCombined.containsKey(str)) {
                    String numValues = freqTableForNominalNumericCombined.get(str);
                    String[] values = numValues.split(",");
                    double attr1 =  inst.value(idx1);
                    for(int i = 0; i < values.length; i++) {
                        double attr11 = Double.valueOf(values[i]);
                        if(Math.abs(attr11-attr1) <= 3*attrStds[idx1])
                            strength++;
                    }
                }
            }
            if(subspaceAttrType[beginAttrIndex] == 4) {//for a nominal_numeric attribute combination
                //  String str = this.getLabelsofJointAttributes(inst, beginAttrIndex,attrOrder,false,false);
                String str = this.getLabelsofMultipleJointAttributes(inst, beginAttrIndex,attrOrder,dim_subspace);
                if(freqTableForNominalNumericCombined.containsKey(str)) {
                    String numValues = freqTableForNominalNumericCombined.get(str);
                    String[] values = numValues.split(",");
                    int idx2 = ((Integer) attrOrder.elementAt((1+beginAttrIndex)% attrOrder.size()));
                    double attr2 = inst.value(idx2);
                    for(int i = 0; i < values.length; i++) {
                        double attr22 = Double.valueOf(values[i]);
                        if(Math.abs(attr22-attr2) <= 3*attrStds[idx2])
                            strength++;
                    }
                }
            }
            return strength;
        }
        
        
        /**
         * to merge postings of two labels
         * @param list1 postings of the forward label
         * @param list2 postings of the post label
         * @return the size of the intersection between two postings
         */
        protected int joinMergeTwoPostings(ArrayList<Integer>list1, ArrayList<Integer>list2) {
            int intersectCounts = 0;
            int len1 = list1.size();
            int len2 = list2.size();
            int i = 0, j = 0;
            while(i < len1 && j < len2) {
                if(list1.get(i) > list2.get(j))
                    j++;
                else if(list1.get(i) < list2.get(j))
                    i++;
                else {
                    intersectCounts++;
                    i++;
                    j++;
                }
            }
            return intersectCounts;
        }
        
    }
    
    protected class ComparatorUtil implements Comparator<Map.Entry>, Serializable {
        
        @Override
        public int compare(Map.Entry o1, Map.Entry o2) {
            return ((Integer) o2.getValue()).compareTo((Integer) o1.getValue());
        }
        
    }
    
    protected class ComparatorUtilAscendRanking implements Comparator<Map.Entry>, Serializable {
        
        @Override
        public int compare(Map.Entry o1, Map.Entry o2) {
            return ((Double) o1.getValue()).compareTo((Double) o2.getValue());
        }
        
    }
}