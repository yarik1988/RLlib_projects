using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.UIElements;
using UnityEngine.Assertions.Must;
using Unity.Barracuda;
using Unity.MLAgents.Policies;
using static UnityEngine.GraphicsBuffer;

public class MissileScript : Agent
{
    public float force = 0.1f;
    public float rot_impulse = 1.0f;
    public float reward_magnitude = 0;
    Rigidbody2D m_Rigidbody;
    LineRenderer m_Renderer;
    public GameObject other_missile;
    public double hit_reward;
    public bool is_right = false;
    public double coins;
    KeyCode left_turn;
    KeyCode right_turn;
    private Camera cur_cam;
    private Rect game_rect;

    public override void Initialize()
    {
        m_Rigidbody = GetComponent<Rigidbody2D>();
        m_Renderer = GetComponent<LineRenderer>();
        cur_cam=transform.parent.gameObject.GetComponent<Camera>();
        game_rect =new Rect(0, 0, 1.0f, 1.0f);
    }

    public override void OnEpisodeBegin()
    {
        m_Rigidbody.velocity = new Vector2(0, 0);
        var x_pos = is_right ? Random.Range(5.0f, 15.0f) : Random.Range(-15.0f, -5.0f);
        if (is_right)
        {
            left_turn = KeyCode.LeftArrow;
            right_turn = KeyCode.RightArrow;
        }
        else
        {
            left_turn = KeyCode.A;
            right_turn = KeyCode.D;
        }
        transform.localPosition = new Vector3(x_pos, Random.Range(-8.0f, 8.0f),-1);
        float angle_to_center=180-Mathf.Atan2(transform.localPosition.x, transform.localPosition.y)* Mathf.Rad2Deg;
        m_Rigidbody.rotation =Random.Range(angle_to_center-30.0f, angle_to_center + 30.0f);
        m_Rigidbody.angularVelocity = 0;
        hit_reward = 0;
        coins = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        add_obs_for_rb(sensor,gameObject);
        add_obs_for_rb(sensor,other_missile);
    }

    void add_obs_for_rb(VectorSensor sensor, GameObject missile)
    {
        Rigidbody2D rb = missile.GetComponent<Rigidbody2D>();
        Vector2 lpos = missile.transform.localPosition;
        sensor.AddObservation(lpos/10.0f);
        sensor.AddObservation(rb.velocity/2.0f);
        sensor.AddObservation(Mathf.Sin(rb.rotation*Mathf.Deg2Rad));
        sensor.AddObservation(Mathf.Cos(rb.rotation*Mathf.Deg2Rad));
        sensor.AddObservation(rb.angularVelocity/50.0f);
    }


    // Update is called once per frame
    void FixedUpdate()
    {
       float angle = -m_Rigidbody.rotation * Mathf.PI / 180.0f;
       Vector2 cur_direction = new Vector2(Mathf.Sin(angle), Mathf.Cos(angle));
       m_Rigidbody.AddForce(force * cur_direction, ForceMode2D.Impulse);
       RaycastHit2D[] hits = Physics2D.RaycastAll(transform.position, cur_direction);
       m_Renderer.enabled = false;     
       foreach (RaycastHit2D hit in hits) 
            if (hit.collider != null && hit.collider.gameObject==other_missile)
            {
                hit_reward+=reward_magnitude;
                other_missile.GetComponent<MissileScript>().hit_reward = -hit_reward;
                m_Renderer.SetPosition(1, new Vector3(0, hit.distance, 0));
                m_Renderer.enabled = true;
                break;
            }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector2 pos=cur_cam.WorldToViewportPoint(transform.position);
        AddReward(0.001f);
        if (!game_rect.Contains(pos))
        {
            AddReward(-1.0f);
            other_missile.GetComponent<MissileScript>().AddReward(1.0f);
            EndEpisode();
            other_missile.GetComponent<MissileScript>().EndEpisode();
        }
        AddReward((float)hit_reward);
        coins += hit_reward;
        hit_reward = 0.0f;
        int sign = actionBuffers.DiscreteActions[0] - 1;
        m_Rigidbody.AddTorque(rot_impulse*sign);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 1;
        if (Input.GetKey(left_turn))
            discreteActionsOut[0] += 1;
        if (Input.GetKey(right_turn))
            discreteActionsOut[0] -= 1;
    }

}
