def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

def run_quickstart():
    # Catalan	ca
    # Spanish	es
    # [START translate_quickstart]
    # Imports the Google Cloud client library
    from google.cloud import translate

    # Instantiates a client
    translate_client = translate.Client()

    # The text to translate
    text1= u'Proposta Consell'
    text2= u'Hi ha dues'
    text3= u'possibilitats'
    text4= u'Tenim un greu'
    # The target language
    source='ca'
    target = 'es'

    # Translates some text into Russian
    translation = translate_client.translate(
        [text1, text2],
        source_language=source,
        target_language=target,
        format_='text'
    )

# translation[0]['input']
# translation[0]['translatedText']




if __name__ == '__main__':
    run_quickstart()
