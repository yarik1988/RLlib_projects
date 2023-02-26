using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.MLAgents.Policies;
using UnityEngine;

public class MissileManagerScript : MonoBehaviour
{
    public GameObject missileLeftPrefab;
    public GameObject missileRightPrefab;
    public GameObject scoreText;
    GameObject left_missile;
    GameObject right_missile;
    public bool is_evaluate=false;
    void Start()
    {
        left_missile=Instantiate(missileLeftPrefab,transform);
        right_missile = Instantiate(missileRightPrefab,transform);
        left_missile.layer = gameObject.layer;
        right_missile.layer = gameObject.layer;
        left_missile.GetComponent<MissileScript>().other_missile = right_missile;
        right_missile.GetComponent<MissileScript>().other_missile = left_missile;
        if (is_evaluate) right_missile.GetComponent<BehaviorParameters>().BehaviorType = BehaviorType.HeuristicOnly;
    }

    // Update is called once per frame
    void Update()
    {
        scoreText.GetComponent<TextMeshPro>().SetText(String.Format("Yellow missile: {0:0.000}, Blue missile: {1:0.000}",
            right_missile.GetComponent<MissileScript>().coins, left_missile.GetComponent<MissileScript>().coins)); 
    }
}
