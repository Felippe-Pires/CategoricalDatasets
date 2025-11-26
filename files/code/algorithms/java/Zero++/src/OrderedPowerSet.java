/*
* To change this license header, choose License Headers in Project Properties.
* To change this template file, choose Tools | Templates
* and open the template in the editor.
*/


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author gpan20
 */
public class OrderedPowerSet<E> {
    private static final int ELEMENT_LIMIT = 80;
    private List<E> inputList;
    public int N;
    private Map<Integer, List<LinkedHashSet<E>>> map =
            new HashMap<Integer, List<LinkedHashSet<E>>>();
    
    public static void  main(String[] args) {
        ArrayList<Integer> t = new ArrayList();
        t.add(1);
        t.add(2);
        t.add(3);
        t.add(4);
        /* t.add(5);
        t.add(6);
        t.add(7);
        t.add(8);
        t.add(9);
       t.add(10);
        t.add(11);
        t.add(12);
        t.add(13);
        t.add(14);
        t.add(15);*/
        OrderedPowerSet ops = new OrderedPowerSet(t);
        for(int j = 1; j <= t.size(); j++) {
            List set = ops.getPermutationsList(j);
            for(int i = 0; i < set.size(); i++) {
                Object o = set.get(i);
               // Object ar[] = ((Set)o).toArray();
                System.out.println(o.toString());
            }
        }
       // System.out.println("size:"+);
    }
    
    public OrderedPowerSet(List<E> list) {
        inputList = list;
        N = list.size();
        if (N > ELEMENT_LIMIT) {
            throw new RuntimeException(
                    "List with more then " + ELEMENT_LIMIT + " elements is too long...");
        }
    }
    
    public List<LinkedHashSet<E>> getPermutationsList(int elementCount) {
        if (elementCount < 1 || elementCount > N) {
            throw new IndexOutOfBoundsException(
                    "Can only generate permutations for a count between 1 to " + N);
        }
        if (map.containsKey(elementCount)) {
            return map.get(elementCount);
        }
        
        ArrayList<LinkedHashSet<E>> list = new ArrayList<LinkedHashSet<E>>();
        
        if (elementCount == N) {
            list.add(new LinkedHashSet<E>(inputList));
        } else if (elementCount == 1) {
            for (int i = 0 ; i < N ; i++) {
                LinkedHashSet<E> set = new LinkedHashSet<E>();
                set.add(inputList.get(i));
                list.add(set);
            }
        } else {
            list = new ArrayList<LinkedHashSet<E>>();
            for (int i = 0 ; i <= N - elementCount ; i++) {
                @SuppressWarnings("unchecked")
                        ArrayList<E> subList = (ArrayList<E>)((ArrayList<E>)inputList).clone();
                for (int j = i ; j >= 0 ; j--) {
                    subList.remove(j);
                }
                OrderedPowerSet<E> subPowerSet =
                        new OrderedPowerSet<E>(subList);
                
                List<LinkedHashSet<E>> pList =
                        subPowerSet.getPermutationsList(elementCount-1);
                for (LinkedHashSet<E> s : pList) {
                    LinkedHashSet<E> set = new LinkedHashSet<E>();
                    set.add(inputList.get(i));
                    set.addAll(s);
                    list.add(set);
                }
            }
        }
        
        map.put(elementCount, list);
        
        return map.get(elementCount);
    }
}