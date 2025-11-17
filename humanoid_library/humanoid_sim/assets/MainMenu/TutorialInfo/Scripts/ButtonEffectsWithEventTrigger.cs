using UnityEngine;

public class ButtonEffectsWithEventTrigger : MonoBehaviour
{
    public Vector3 hoverScale = new Vector3(1.1f, 1.1f, 1.1f);
    public Vector3 clickScale = new Vector3(1.2f, 1.2f, 1.2f);
    private Vector3 originalScale;

    private void Start()
    {
        originalScale = transform.localScale;
    }

    public void OnHoverEnter()
    {
        transform.localScale = hoverScale;
    }

    public void OnHoverExit()
    {
        transform.localScale = originalScale;
    }

    public void OnClick()
    {
        transform.localScale = clickScale;
        Invoke(nameof(ResetScale), 0.1f); // Reset scale after a short delay
    }

    private void ResetScale()
    {
        transform.localScale = originalScale;
    }
}
