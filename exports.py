from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import date, timedelta, datetime

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import os, io, json, base64
from sqlalchemy import create_engine
import pandas as pd

SECRET_ROOT = '../../SECRETS/'

class EmailHandler:
    def __init__(self):
        SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]

        local_key_loc = SECRET_ROOT + 'gmail_token.json'
        email_list_loc = SECRET_ROOT + 'email_list.json'

        if os.path.exists(local_key_loc): # for local testing
            creds = Credentials.from_authorized_user_file(local_key_loc, SCOPES)
            with open(email_list_loc) as f:
                email_list = json.load(f)
                SENDER = email_list['SENDER']
                RECIPIENTS = email_list['RECIPIENTS']
        else:
            tmp = os.environ['gmail_token']
            secret_value = json.loads(tmp)
            creds = Credentials.from_authorized_user_info(secret_value)
            SENDER = os.environ['SENDER']
            RECIPIENTS = os.environ['RECIPIENTS'].split(',')


        self.service = build("gmail", "v1", credentials=creds)
        self.date_str = date.today().strftime('%Y/%m/%d')
        self.recipients = RECIPIENTS
        self.to_str = ', '.join(self.recipients)
        self.sender = SENDER
        self.encoded_email = None

    def draft_msg(self, subject=None, text=None, images=None):

        if subject is None:
            subject = f'MSG - {self.date_str}'
        if text is None:
            text = f'Predictions for {self.date_str}'

        message = MIMEMultipart()
        message['to'] = self.to_str
        message['from'] = self.sender
        message['subject'] = subject

        html_part = MIMEText(text)
        message.attach(html_part)

        if images is not None:
            if type(images) is str:
                images = [images]
            for image in images:
                with open(image, 'rb') as f:
                    image_part = MIMEImage(f.read())
                message.attach(image_part)
        self.encoded_email = base64.urlsafe_b64encode(message.as_bytes()).decode()

    def send_msg(self):
        self.service.users().messages().send(userId='me', body={'raw': self.encoded_email}).execute()

class SQLHandler:
    def __init__(self):
        key_loc = SECRET_ROOT + 'SQL_token.json'
        # otherwise will be held in secrets manager
        if os.path.exists(key_loc):
            with open(key_loc) as f:
                d = json.load(f)
        else:
            tmp = os.environ['sql_token']
            d = json.loads(tmp)
        self.engine = create_engine(d['engine_url'])

    def write_new_table(self, df, table_name, overwrite = False):
        connection = self.engine.raw_connection()

        if overwrite:
            df.head(0).to_sql(table_name, self.engine, if_exists='replace', index=False)
        cur = connection.cursor()
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        cur.copy_from(output, table_name, null="")  # null values become ''
        connection.commit()
        cur.close()
        connection.close()

    def read_table(self,table_name, add_sql=''):
        connection = self.engine.connect()
        df = pd.read_sql(f"SELECT * FROM {table_name}" + add_sql, self.engine)
        connection.close()
        return df


