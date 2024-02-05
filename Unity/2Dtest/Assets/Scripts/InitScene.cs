using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Policies;
using UnityEngine;

public class InitScene : MonoBehaviour
{
    // Start is called before the first frame update
    [SerializeField]
    public NNModel modelAsset;
    public GameObject AgentPrefab;
    void Start()
    {
        int num_agents = (modelAsset == null) ? 1 : 4;
        for (int i=0;i<num_agents;i++)
            {
            GameObject cur_agent = Instantiate(AgentPrefab,new Vector2(0,0), Quaternion.identity);
            BehaviorParameters bp = cur_agent.GetComponent<BehaviorParameters>();
            bp.Model = modelAsset;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
