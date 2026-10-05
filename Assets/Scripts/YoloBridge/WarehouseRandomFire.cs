using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WarehouseRandomFire : MonoBehaviour
{
    public GameObject firePrefab;
    public Transform playerHead;
    public Transform[] spawnPoints;
    public Vector2 spawnInterval = new Vector2(8, 16);
    public Vector2 duration = new Vector2(6, 12);
    [Min(1)] public int maximumSimultaneousFires = 2;
    public float playerClearance = 2;
    public float obstacleRadius = .65f;
    public float fireScale = 1;
    public bool spawningEnabled = true;
    // Validation only. YOLO bridge never reads these states.
    public int TotalSpawned { get; private set; }
    public int TotalExpired { get; private set; }
    public int ActiveFireCount { get { return fires.Count; } }
    readonly Dictionary<GameObject, float> fires = new Dictionary<GameObject, float>();
    readonly List<GameObject> expired = new List<GameObject>();
    float nextSpawn;
    void OnEnable() { nextSpawn = Time.time + 2; }
    void Update()
    {
        expired.Clear();
        foreach (var pair in fires)
            if (!pair.Key || Time.time >= pair.Value) expired.Add(pair.Key);
        foreach (var fire in expired) { fires.Remove(fire); if (fire) { Destroy(fire); TotalExpired++; } }
        if (!spawningEnabled || Time.time < nextSpawn) return;
        nextSpawn = Time.time + Random.Range(Mathf.Max(.2f, spawnInterval.x), Mathf.Max(.2f, spawnInterval.y));
        TrySpawn();
    }
    public bool TrySpawn()
    {
        if (!firePrefab || spawnPoints == null || spawnPoints.Length == 0 ||
            fires.Count >= Mathf.Max(1, maximumSimultaneousFires)) return false;
        int offset = Random.Range(0, spawnPoints.Length);
        for (int i = 0; i < spawnPoints.Length; i++)
        {
            var point = spawnPoints[(offset + i) % spawnPoints.Length];
            if (!point) continue;
            RaycastHit hit;
            if (!Physics.Raycast(point.position + Vector3.up * 2, Vector3.down, out hit, 4,
                                 ~0, QueryTriggerInteraction.Ignore) || hit.normal.y < .95f ||
                Mathf.Abs(hit.point.y - point.position.y) > .2f) continue;
            var position = hit.point + Vector3.up * .03f;
            if (playerHead && Vector2.Distance(new Vector2(position.x, position.z),
                new Vector2(playerHead.position.x, playerHead.position.z)) < playerClearance) continue;
            bool occupied = false;
            foreach (var existing in fires.Keys)
                if (existing && Vector3.Distance(existing.transform.position, position) < obstacleRadius * 2) occupied = true;
            foreach (var collider in Physics.OverlapCapsule(position + Vector3.up * (obstacleRadius + .1f),
                position + Vector3.up * 1.9f, obstacleRadius, ~0, QueryTriggerInteraction.Ignore))
                occupied = true;
            // Animated people can have no Collider. Protect their bodies using rendered bounds as well.
            foreach (var person in FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None))
                if (person.name == "Body" && person.bounds.SqrDistance(position + Vector3.up) < 1) occupied = true;
            if (occupied) continue;
            var fire = Instantiate(firePrefab, position, Quaternion.identity, transform);
            fire.name = "Random Fire VFX";
            fire.AddComponent<WarehouseExtinguishableFire>();
            fire.transform.localScale = Vector3.one * Mathf.Max(.1f, fireScale);
            foreach (var particles in fire.GetComponentsInChildren<ParticleSystem>()) particles.Play(true);
            fires.Add(fire, Time.time + Random.Range(Mathf.Max(.2f, duration.x), Mathf.Max(.2f, duration.y)));
            TotalSpawned++;
            return true;
        }
        return false;
    }
    public void ClearFires()
    {
        foreach (var fire in fires.Keys) if (fire) Destroy(fire);
        fires.Clear();
    }
    void OnDisable() { ClearFires(); }
}