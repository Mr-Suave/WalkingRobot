using UnityEngine;

public class ExitButton : MonoBehaviour
{
    // Call this method when the exit button is clicked
    public void ExitGame()
    {
        // Log a message for debugging in the editor
        Debug.Log("Exiting the game...");

        // Quit the application
        Application.Quit();

        // If running in the editor, stop play mode
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }
}
