using UnityEngine;

[DisallowMultipleComponent]
public sealed class AutoRotate : MonoBehaviour
{
    private const float DegreesPerSecond = 45f;

    private void Update()
    {
        transform.Rotate(Vector3.up, DegreesPerSecond * Time.deltaTime, Space.World);
    }
}
