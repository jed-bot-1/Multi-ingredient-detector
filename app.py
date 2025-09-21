const sendToAPI = async (photo) => {
    try {
        const formData = new FormData();
        
        // Option 1 (easiest):
        formData.append('image', {
            uri: photo.uri,
            type: 'image/jpeg',
            name: 'photo.jpg'
        } as any);

        const response = await fetch('https://multi-ingredient-detector.onrender.com/detect/', {
            method: 'POST',
            body: formData,
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Success:', result);
        }
    } catch (error) {
        console.error('Error:', error);
    }
};
