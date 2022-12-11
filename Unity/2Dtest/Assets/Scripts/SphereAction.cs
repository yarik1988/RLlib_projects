using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;

public class SphereAction : Agent
{
    public float impulseStep = 1.0f;
    Rigidbody2D m_Rigidbody;
    Vector2 init_pos;
    bool is_finish;

    public override void Initialize()
    {
        m_Rigidbody = GetComponent<Rigidbody2D>();
        init_pos = this.transform.localPosition;
    }

    public override void OnEpisodeBegin()
    {
        this.transform.localPosition = init_pos;
        m_Rigidbody.velocity = new Vector2(0, 0);
        m_Rigidbody.rotation = 0;
        is_finish = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(m_Rigidbody.position);
        sensor.AddObservation(m_Rigidbody.velocity);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (is_finish)
        {
            AddReward(1.0f);
            EndEpisode();
        }
        else
        {
            Vector3 controlSignal = Vector3.zero;
            int sign = actionBuffers.DiscreteActions[0] - 1;
            m_Rigidbody.AddForce(new Vector2(impulseStep * sign, 0), ForceMode2D.Impulse);
            AddReward(-0.001f);
        }
    }

    public void OnTriggerExit2D(Collider2D collider)
    {
        is_finish = true;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 1;
        if (Input.GetKey(KeyCode.LeftArrow))
            discreteActionsOut[0] -= 1;
        if (Input.GetKey(KeyCode.RightArrow))
            discreteActionsOut[0] += 1;
    }



}
